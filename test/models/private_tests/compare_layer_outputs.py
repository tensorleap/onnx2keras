import argparse
import logging
from collections import defaultdict
import copy

import numpy as np
import onnx
import onnxruntime as ort
from keras.models import Model

from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last


def _make_input_arrays(session, seed=42):
    rng = np.random.default_rng(seed=seed)
    input_arrays = {}
    input_names = []
    for input_info in session.get_inputs():
        input_name = input_info.name
        input_names.append(input_name)
        input_shape = input_info.shape
        test_shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim is None or dim == -1:
                test_shape.append(1)
            else:
                test_shape.append(dim)
        input_arrays[input_name] = rng.random(test_shape).astype(np.float32)
    return input_names, input_arrays


def _collect_onnx_outputs(onnx_model, input_arrays, target_ops, max_per_type, ordered_outputs):
    model = copy.deepcopy(onnx_model)
    inferred = onnx.shape_inference.infer_shapes(model)
    value_info_map = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
        value_info_map[vi.name] = vi

    output_names = []
    output_types = []
    op_type_map = {}
    for node in onnx_model.graph.node:
        for out_name in node.output:
            if out_name:
                op_type_map[out_name] = node.op_type

    op_counts = defaultdict(int)
    for out_name in ordered_outputs:
        op_type = op_type_map.get(out_name)
        if op_type is None or op_type not in target_ops:
            continue
        if max_per_type and op_counts[op_type] >= max_per_type:
            continue
        vi = value_info_map.get(out_name)
        if vi is None:
            continue
        if all(o.name != out_name for o in model.graph.output):
            model.graph.output.append(vi)
        output_names.append(out_name)
        output_types.append(op_type)
        op_counts[op_type] += 1

    session = ort.InferenceSession(model.SerializeToString())
    outputs = session.run(output_names, input_arrays)
    onnx_by_type = defaultdict(list)
    for op_type, out in zip(output_types, outputs):
        onnx_by_type[op_type].append(out)
    return onnx_by_type


def _collect_keras_outputs(keras_model, input_arrays, target_ops, max_per_type, ordered_outputs, tensor_map):
    outputs = []
    layers = []
    op_counts = defaultdict(int)
    layer_by_name = {layer.name: layer for layer in keras_model.layers}
    for out_name in ordered_outputs:
        keras_name = tensor_map.get(out_name)
        if not keras_name:
            continue
        layer = layer_by_name.get(keras_name)
        if layer is None:
            continue
        mapped = _map_layer_type(layer.__class__.__name__)
        if mapped is None or mapped not in target_ops:
            continue
        if max_per_type and op_counts[mapped] >= max_per_type:
            continue
        if hasattr(layer, "output"):
            outputs.append(layer.output)
            layers.append(layer)
            op_counts[mapped] += 1
    intermediate_model = Model(inputs=keras_model.inputs, outputs=outputs)
    keras_outputs = intermediate_model([input_arrays[name] for name in keras_model.input_names])

    keras_by_type = defaultdict(list)
    for layer, out in zip(layers, keras_outputs):
        if isinstance(out, (list, tuple)):
            if len(out) == 0:
                continue
            out = out[0]
        keras_by_type[layer.__class__.__name__].append(out.numpy())
    return keras_by_type


def _collect_onnx_tensor_values(onnx_model, input_arrays, tensor_names):
    model = copy.deepcopy(onnx_model)
    inferred = onnx.shape_inference.infer_shapes(model)
    value_info_map = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
        value_info_map[vi.name] = vi

    output_names = []
    for name in tensor_names:
        vi = value_info_map.get(name)
        if vi is None:
            continue
        if all(o.name != name for o in model.graph.output):
            model.graph.output.append(vi)
        output_names.append(name)

    session = ort.InferenceSession(model.SerializeToString())
    outputs = session.run(output_names, input_arrays)
    return {name: out for name, out in zip(output_names, outputs)}


def _build_keras_tensor_map(keras_model, tensor_map):
    keras_tensor_map = {}
    for onnx_name, keras_layer_name in tensor_map.items():
        try:
            layer = keras_model.get_layer(keras_layer_name)
        except Exception:
            continue
        if hasattr(layer, "output"):
            keras_tensor_map[onnx_name] = layer.output
    for input_tensor in keras_model.inputs:
        input_name = input_tensor.name.split(":")[0]
        try:
            layer = keras_model.get_layer(input_name)
            keras_tensor_map[input_name] = layer.output
        except Exception:
            keras_tensor_map[input_name] = input_tensor
    return keras_tensor_map


def compare_isolated_nodes(onnx_model, keras_model, input_arrays, target_ops, max_per_type, tensor_map):
    op_counts = defaultdict(int)
    required_tensors = set()
    nodes_to_check = []
    init_names = {init.name for init in onnx_model.graph.initializer}

    for node in onnx_model.graph.node:
        if node.op_type not in target_ops:
            continue
        if max_per_type and op_counts[node.op_type] >= max_per_type:
            continue
        nodes_to_check.append(node)
        op_counts[node.op_type] += 1
        required_tensors.update([name for name in node.input if name and name not in init_names])
        required_tensors.update([name for name in node.output if name])

    onnx_values = _collect_onnx_tensor_values(onnx_model, input_arrays, sorted(required_tensors))
    keras_tensor_map = _build_keras_tensor_map(keras_model, tensor_map)

    results = []
    for node in nodes_to_check:
        if len(node.output) == 0:
            continue
        out_name = node.output[0]
        if out_name not in onnx_values or out_name not in keras_tensor_map:
            continue

        input_tensors = []
        input_values = []
        missing = False
        for inp in node.input:
            if inp in init_names:
                continue
            if inp in keras_tensor_map and inp in onnx_values:
                input_tensors.append(keras_tensor_map[inp])
                input_values.append(_align_to_keras(onnx_values[inp], keras_tensor_map[inp]))
            else:
                missing = True
                break
        if missing or not input_tensors:
            continue

        submodel = Model(inputs=input_tensors, outputs=keras_tensor_map[out_name])
        keras_out = submodel(input_values)
        if isinstance(keras_out, (list, tuple)):
            if len(keras_out) == 0:
                continue
            keras_out = keras_out[0]
        keras_np = keras_out.numpy() if hasattr(keras_out, "numpy") else keras_out
        onnx_out = _align_to_keras(onnx_values[out_name], keras_tensor_map[out_name])

        if keras_np.shape != onnx_out.shape:
            continue

        diff = np.abs(keras_np - onnx_out)
        results.append({
            "op_type": node.op_type,
            "onnx_output": out_name,
            "mean_error": float(diff.mean()),
            "max_error": float(diff.max()),
        })

    results.sort(key=lambda x: x["mean_error"], reverse=True)
    return results


def _map_layer_type(layer_type):
    mapping = {
        "Conv1D": "Conv",
        "Conv2D": "Conv",
        "Conv3D": "Conv",
        "Conv2DTranspose": "ConvTranspose",
        "Conv3DTranspose": "ConvTranspose",
        "GroupedConvTranspose": "ConvTranspose",
        "Add": "Add",
        "Multiply": "Mul",
        "Reshape": "Reshape",
        "Concatenate": "Concat",
        "ZeroPadding2D": "Pad",
        "ZeroPadding3D": "Pad",
        "Cropping2D": "Crop",
        "Cropping3D": "Crop",
    }
    return mapping.get(layer_type)


def _align_to_keras(value, keras_tensor):
    if not hasattr(keras_tensor, "shape"):
        return value
    keras_shape = tuple(keras_tensor.shape)
    if value is None or keras_shape is None:
        return value
    if len(value.shape) != len(keras_shape):
        return value
    if len(value.shape) == 4:
        if value.shape[1] == keras_shape[-1] and value.shape[2:] == keras_shape[1:-1]:
            return np.transpose(value, (0, 2, 3, 1))
    if len(value.shape) == 5:
        if value.shape[1] == keras_shape[-1] and value.shape[2:] == keras_shape[1:-1]:
            return np.transpose(value, (0, 2, 3, 4, 1))
    return value


def compare_by_type(onnx_by_type, keras_by_type):
    results = []
    for keras_type, keras_outputs in keras_by_type.items():
        onnx_type = _map_layer_type(keras_type)
        if onnx_type is None or onnx_type not in onnx_by_type:
            continue
        onnx_outputs = onnx_by_type[onnx_type]
        pair_count = min(len(onnx_outputs), len(keras_outputs))
        if pair_count == 0:
            continue
        mean_errors = []
        max_errors = []
        matched = 0
        for i in range(pair_count):
            onnx_out = onnx_outputs[i]
            keras_out = keras_outputs[i]
            if onnx_out.shape != keras_out.shape:
                continue
            diff = np.abs(onnx_out - keras_out)
            mean_errors.append(diff.mean())
            max_errors.append(diff.max())
            matched += 1
        if matched:
            results.append({
                "onnx_type": onnx_type,
                "keras_type": keras_type,
                "count": matched,
                "mean_error": float(np.mean(mean_errors)),
                "max_error": float(np.max(max_errors)),
            })
    results.sort(key=lambda x: x["mean_error"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare ONNX vs Keras outputs by layer type.")
    parser.add_argument("--model", default="test/models/private_tests/asensus/lung_anatomy_merged_ir9.onnx")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ops", default="ConvTranspose", help="Comma-separated ONNX op types to compare, or 'all'")
    parser.add_argument("--max-per-type", type=int, default=10)
    parser.add_argument("--isolated", action="store_true", help="Compare each node in isolation using ONNX inputs")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)

    onnx_model = onnx.load(args.model)
    session = ort.InferenceSession(args.model)
    input_names, input_arrays = _make_input_arrays(session, seed=args.seed)

    response = onnx_to_keras(
        onnx_model,
        input_names,
        name_policy="attach_weights_name",
        allow_partial_compilation=False,
        verbose=False,
        return_tensor_map=True,
    )
    keras_model = response.converted_model
    tensor_map = response.tensor_map
    if not tensor_map:
        print("No tensor map returned; cannot run exact comparison.")
        return

    ordered_outputs = [name for name in tensor_map.keys() if name]
    if args.ops.strip().lower() == "all":
        target_ops = {node.op_type for node in onnx_model.graph.node}
    else:
        target_ops = {op.strip() for op in args.ops.split(",") if op.strip()}
    final_model = convert_channels_first_to_last(
        keras_model,
        should_transform_inputs_and_outputs=False
    )
    if args.isolated:
        results = compare_isolated_nodes(
            onnx_model,
            final_model,
            input_arrays,
            target_ops,
            args.max_per_type,
            tensor_map,
        )
    else:
        onnx_by_type = _collect_onnx_outputs(
            onnx_model,
            input_arrays,
            target_ops,
            args.max_per_type,
            ordered_outputs,
        )
        keras_by_type = _collect_keras_outputs(
            final_model,
            input_arrays,
            target_ops,
            args.max_per_type,
            ordered_outputs,
            tensor_map,
        )
        results = compare_by_type(onnx_by_type, keras_by_type)
    if not results:
        print("No comparable layer types found.")
        return

    if args.isolated:
        if args.ops.strip().lower() == "all":
            by_op = {}
            for r in results:
                by_op[r["op_type"]] = r
            print("Isolated-node mismatches (one per op type):")
            for op_type in sorted(by_op.keys()):
                r = by_op[op_type]
                print(
                    f"{op_type}: mean_error={r['mean_error']:.6e}, "
                    f"max_error={r['max_error']:.6e}, output={r['onnx_output']}"
                )
        else:
            print("Top isolated-node mismatches (mean_error desc):")
            for r in results[:10]:
                print(
                    f"{r['op_type']} {r['onnx_output']}: "
                    f"mean_error={r['mean_error']:.6e}, "
                    f"max_error={r['max_error']:.6e}"
                )
    else:
        print("Top layer-type mismatches (mean_error desc):")
        for r in results[:10]:
            print(
                f"{r['onnx_type']} vs {r['keras_type']}: "
                f"count={r['count']}, mean_error={r['mean_error']:.6e}, "
                f"max_error={r['max_error']:.6e}"
            )


if __name__ == "__main__":
    main()

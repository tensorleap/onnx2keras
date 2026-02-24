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


def _collect_onnx_outputs(onnx_model, input_arrays, target_ops, max_per_type):
    model = copy.deepcopy(onnx_model)
    inferred = onnx.shape_inference.infer_shapes(model)
    value_info_map = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
        value_info_map[vi.name] = vi

    output_names = []
    output_types = []
    op_counts = defaultdict(int)
    for node in onnx_model.graph.node:
        if node.op_type not in target_ops:
            continue
        if max_per_type and op_counts[node.op_type] >= max_per_type:
            continue
        for out_name in node.output:
            if out_name:
                vi = value_info_map.get(out_name)
                if vi is None:
                    continue
                if all(o.name != out_name for o in model.graph.output):
                    model.graph.output.append(vi)
                output_names.append(out_name)
                output_types.append(node.op_type)
        op_counts[node.op_type] += 1

    session = ort.InferenceSession(model.SerializeToString())
    outputs = session.run(output_names, input_arrays)
    onnx_by_type = defaultdict(list)
    for op_type, out in zip(output_types, outputs):
        onnx_by_type[op_type].append(out)
    return onnx_by_type


def _collect_keras_outputs(keras_model, input_arrays, target_ops, max_per_type):
    outputs = []
    layers = []
    op_counts = defaultdict(int)
    for layer in keras_model.layers:
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
    parser.add_argument("--ops", default="ConvTranspose", help="Comma-separated ONNX op types to compare")
    parser.add_argument("--max-per-type", type=int, default=10)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)

    onnx_model = onnx.load(args.model)
    session = ort.InferenceSession(args.model)
    input_names, input_arrays = _make_input_arrays(session, seed=args.seed)

    target_ops = {op.strip() for op in args.ops.split(",") if op.strip()}
    onnx_by_type = _collect_onnx_outputs(onnx_model, input_arrays, target_ops, args.max_per_type)

    keras_model = onnx_to_keras(
        onnx_model,
        input_names,
        name_policy="attach_weights_name",
        allow_partial_compilation=False,
        verbose=False,
    ).converted_model
    final_model = convert_channels_first_to_last(
        keras_model,
        should_transform_inputs_and_outputs=False
    )
    keras_by_type = _collect_keras_outputs(final_model, input_arrays, target_ops, args.max_per_type)

    results = compare_by_type(onnx_by_type, keras_by_type)
    if not results:
        print("No comparable layer types found.")
        return

    print("Top layer-type mismatches (mean_error desc):")
    for r in results[:10]:
        print(
            f"{r['onnx_type']} vs {r['keras_type']}: "
            f"count={r['count']}, mean_error={r['mean_error']:.6e}, "
            f"max_error={r['max_error']:.6e}"
        )


if __name__ == "__main__":
    main()

import time

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2kerastl import onnx_to_keras
from onnx2kerastl.utils import LARGE_CONSTANT_THRESHOLD


def _build_model(data_shape, axis):
    # GatherElements with a fully-defined indices shape, like BEVFusion's
    # TopK-derived peak gathers -- data/indices are both runtime tensors (not
    # constants), matching how the real model feeds this op.
    nodes = [
        helper.make_node('Cast', ['indices'], ['indices_i'], to=TensorProto.INT64),
        helper.make_node('GatherElements', ['data', 'indices_i'], ['y'], axis=axis),
    ]
    graph = helper.make_graph(
        nodes, 'gather_elements_static',
        [helper.make_tensor_value_info('data', TensorProto.FLOAT, list(data_shape)),
         helper.make_tensor_value_info('indices', TensorProto.FLOAT, list(data_shape))],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, list(data_shape))],
        [])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])


def test_gather_elements_large_static_shape_not_embedded():
    # (1, 640, 480): batch axis (1) stripped by onnx_to_keras's Input
    # convention, so the traced indices KerasTensor has a fully-defined static
    # shape (640, 480) -- ~300K elements, matching the scale of the real
    # BEVFusion GatherElements nodes that triggered this bug. get_config()
    # alone stays cheap even with the bug present (it can hold raw
    # ndarray/EagerTensor refs uncoverted) -- the actual blow-up is in
    # to_json(), which materializes them via json_utils.get_json_type's
    # obj.tolist() fallback, so that's what must be exercised here.
    shape = (1, 640, 480)
    axis = 2
    model = _build_model(shape, axis)

    rng = np.random.default_rng(0)
    data = rng.standard_normal(shape).astype(np.float32)
    indices = rng.integers(0, shape[axis], size=shape).astype(np.float32)

    converted = onnx_to_keras(model, input_names=['data', 'indices'],
                              name_policy='attach_weights_name',
                              allow_partial_compilation=False)
    keras_model = converted.converted_model
    out = np.array(keras_model([data, indices]))

    ref = np.take_along_axis(data, indices.astype(np.int64), axis=axis)
    assert out.shape == ref.shape == shape
    assert np.allclose(out, ref)

    t0 = time.time()
    json_str = keras_model.to_json()
    elapsed = time.time() - t0
    assert elapsed < 10.0, f"to_json() took {elapsed:.1f}s -- large constant likely baked into config"
    assert len(json_str) < 200_000, f"to_json() output is {len(json_str)} bytes -- large constant likely embedded"


def _build_model_constant_data(data_grid, gather_axis, indices_shape):
    # GatherElements with a large CONSTANT data operand (e.g. a fixed
    # coordinate grid, like BEVFusion's dense_heatmap pixel-center lookup) and
    # a runtime indices operand -- this is the pattern the constant-data fix
    # targets (distinct from the large-indices case above).
    data_const = numpy_helper.from_array(data_grid.astype(np.float32), 'data')
    nodes = [
        helper.make_node('Cast', ['indices'], ['indices_i'], to=TensorProto.INT64),
        helper.make_node('GatherElements', ['data', 'indices_i'], ['y'], axis=gather_axis),
    ]
    graph = helper.make_graph(
        nodes, 'gather_elements_constant_data',
        [helper.make_tensor_value_info('indices', TensorProto.FLOAT, list(indices_shape))],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, list(indices_shape))],
        [data_const])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])


def test_gather_elements_large_constant_data_not_embedded():
    # data: a fixed (1, 20000, 2) coordinate-style grid -- 40,000 elements,
    # well past LARGE_CONSTANT_THRESHOLD. indices: a small runtime tensor
    # picking rows out of the grid along axis 1.
    data_shape = (1, 20000, 2)
    assert np.prod(data_shape) > LARGE_CONSTANT_THRESHOLD
    indices_shape = (1, 5, 2)
    gather_axis = 1

    rng = np.random.default_rng(2)
    data_grid = rng.standard_normal(data_shape).astype(np.float32)
    row_picks = rng.integers(0, data_shape[gather_axis], size=(1, 5, 1))
    indices = np.broadcast_to(row_picks, indices_shape).astype(np.float32)

    model = _build_model_constant_data(data_grid, gather_axis, indices_shape)
    converted = onnx_to_keras(model, input_names=['indices'],
                              name_policy='attach_weights_name',
                              allow_partial_compilation=False)
    keras_model = converted.converted_model
    out = np.array(keras_model([indices]))

    ref = np.take_along_axis(data_grid, indices.astype(np.int64), axis=gather_axis)
    assert out.shape == ref.shape == indices_shape
    assert np.allclose(out, ref)

    assert any(type(layer).__name__ == 'OnnxConstant' for layer in keras_model.layers)

    t0 = time.time()
    json_str = keras_model.to_json()
    elapsed = time.time() - t0
    assert elapsed < 10.0, f"to_json() took {elapsed:.1f}s -- large constant likely baked into config"
    assert len(json_str) < 200_000, f"to_json() output is {len(json_str)} bytes -- large constant likely embedded"


def test_gather_elements_coordinate_operands_have_distinct_tensor_names():
    # rank >= 3 leaves two non-gather axes, so torch_gather emits two identity-grid
    # coordinate tensors. tf.where already returns int64, so casting them to int64 is
    # a no-op that produces no TF op -- both KerasTensors then fall back to the
    # generic name "Placeholder:0". The h5 wires the stack by LAYER name and stays
    # correct, but consumers that wire by TENSOR name collapse the duplicate key and
    # feed one axis' coordinates into both slots, so GatherNd indexes the wrong axis.
    data_shape = (1, 32, 2)
    indices_shape = (1, 4, 2)
    gather_axis = 1

    rng = np.random.default_rng(7)
    data = rng.standard_normal(data_shape).astype(np.float32)
    picks = rng.integers(0, data_shape[gather_axis], size=indices_shape).astype(np.float32)

    model = _build_model_constant_data(data, gather_axis, indices_shape)
    keras_model = onnx_to_keras(model, input_names=['indices'],
                                name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model

    # substring, not endswith: keras uniquifies repeated layer names across tests
    # sharing a process, so this can come out as "..._gather_indices_1".
    stack = next(l for l in keras_model.layers if '_gather_indices' in l.name)
    operand_names = [t.name for t in stack.input]
    assert len(set(operand_names)) == len(operand_names), (
        f"stack operands must have unique tensor names, got {operand_names}")
    assert not any(name.startswith('Placeholder') for name in operand_names), (
        f"stack operands must not fall back to the generic placeholder name: {operand_names}")

    out = np.array(keras_model([picks]))
    ref = np.take_along_axis(data, picks.astype(np.int64), axis=gather_axis)
    assert np.allclose(out, ref)

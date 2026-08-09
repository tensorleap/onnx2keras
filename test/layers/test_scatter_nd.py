import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx2kerastl import onnx_to_keras
from onnx2kerastl.utils import LARGE_CONSTANT_THRESHOLD


def _build_model(canvas_shape, n_updates):
    # zero-filled scatter canvas (e.g. a BEV heatmap), like BEVFusion's
    # dense_heatmap ScatterND -- data is a constant, indices/updates are runtime.
    # indices/updates carry a leading size-1 axis that onnx_to_keras treats as
    # the batch dim (its own convention: an Input's first axis is always
    # batch) and Squeeze it away before ScatterND, matching how the real
    # exported model feeds ScatterND from batched upstream tensors.
    canvas = numpy_helper.from_array(
        np.zeros(canvas_shape, dtype=np.float32), 'canvas')
    squeeze_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), 'squeeze_axes')
    nodes = [
        helper.make_node('Squeeze', ['indices', 'squeeze_axes'], ['indices_sq']),
        helper.make_node('Squeeze', ['updates', 'squeeze_axes'], ['updates_sq']),
        helper.make_node('Cast', ['indices_sq'], ['indices_i'], to=TensorProto.INT64),
        helper.make_node('ScatterND', ['canvas', 'indices_i', 'updates_sq'], ['y']),
    ]
    graph = helper.make_graph(
        nodes, 'scatter_nd_canvas',
        [helper.make_tensor_value_info('indices', TensorProto.FLOAT, [1, n_updates, 2]),
         helper.make_tensor_value_info('updates', TensorProto.FLOAT, [1, n_updates])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, list(canvas_shape))],
        [canvas, squeeze_axes])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])


def _run(model, indices, updates):
    converted = onnx_to_keras(model, input_names=['indices', 'updates'],
                              name_policy='attach_weights_name',
                              allow_partial_compilation=False)
    keras_model = converted.converted_model
    out = np.array(keras_model([indices, updates]))

    import onnxruntime as ort
    ref = ort.InferenceSession(model.SerializeToString()).run(
        None, {'indices': indices, 'updates': updates})[0]
    return keras_model, out, ref


def test_scatter_nd_small_canvas_unchanged():
    # below LARGE_CONSTANT_THRESHOLD -- still baked in literally, correctness only
    shape = (10, 10)
    assert np.prod(shape) <= LARGE_CONSTANT_THRESHOLD
    model = _build_model(shape, n_updates=4)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 10, size=(1, 4, 2)).astype(np.float32)
    updates = rng.standard_normal((1, 4)).astype(np.float32)

    _, out, ref = _run(model, indices, updates)
    assert out.shape == ref.shape == shape
    assert np.allclose(out, ref)


def test_scatter_nd_large_zero_canvas_not_embedded():
    # above LARGE_CONSTANT_THRESHOLD -- must route through the weight-backed
    # OnnxConstant path instead of being baked as a literal graph constant
    shape = (120, 120)
    assert np.prod(shape) > LARGE_CONSTANT_THRESHOLD
    model = _build_model(shape, n_updates=5)
    rng = np.random.default_rng(1)
    indices = rng.integers(0, 120, size=(1, 5, 2)).astype(np.float32)
    updates = rng.standard_normal((1, 5)).astype(np.float32)

    keras_model, out, ref = _run(model, indices, updates)
    assert out.shape == ref.shape == shape
    assert np.allclose(out, ref)

    assert any(type(layer).__name__ == 'OnnxConstant' for layer in keras_model.layers)

    t0 = time.time()
    config = keras_model.get_config()
    assert time.time() - t0 < 5.0

    def max_leaf_count(obj, depth=0):
        if depth > 20:
            return 0
        if isinstance(obj, (list, tuple)):
            if obj and isinstance(obj[0], (list, tuple)):
                return len(obj) * max_leaf_count(obj[0], depth + 1)
            return len(obj)
        if isinstance(obj, dict):
            return max((max_leaf_count(v, depth + 1) for v in obj.values()), default=0)
        return 0

    for layer_cfg in config['layers']:
        assert max_leaf_count(layer_cfg) <= LARGE_CONSTANT_THRESHOLD

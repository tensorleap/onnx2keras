import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2kerastl import onnx_to_keras

N = 150


def _build_model():
    rng = np.random.RandomState(0)
    mask = np.tril(np.ones((N, N), dtype=bool))
    fill = rng.randn(N, N).astype(np.float32)
    scale = rng.rand(N, N).astype(np.float32)
    nodes = [
        helper.make_node('Greater', ['x', 'half'], ['gt']),
        helper.make_node('And', ['gt', 'mask'], ['both']),
        helper.make_node('Where', ['both', 'x', 'fill'], ['picked']),
        helper.make_node('Expand', ['picked', 'shape'], ['expanded']),
        helper.make_node('Mul', ['expanded', 'scale'], ['y']),
    ]
    inits = [
        numpy_helper.from_array(np.float32(0.5), 'half'),
        numpy_helper.from_array(mask, 'mask'),
        numpy_helper.from_array(fill, 'fill'),
        numpy_helper.from_array(scale, 'scale'),
        numpy_helper.from_array(np.array([1, N, N], dtype=np.int64), 'shape'),
    ]
    graph = helper.make_graph(
        nodes, 'large_const_ops',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, N, N])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, N, N])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    return model


def _count(x):
    return sum(_count(i) for i in x) if isinstance(x, list) else 1


def _inline_lists(obj, found):
    if isinstance(obj, list) and len(obj) > 8 and _count(obj) > 5000:
        found.append(_count(obj))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _inline_lists(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _inline_lists(v, found)


def test_large_constants_in_and_where_mul_are_not_inlined():
    model = _build_model()
    keras_model = onnx_to_keras(model, input_names=['x'], name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model

    found = []
    for layer in keras_model.get_config()['layers']:
        _inline_lists(layer, found)
    assert found == [], f'large constants baked inline into layer configs: {found}'

    x = np.random.RandomState(1).randn(1, N, N).astype(np.float32)
    out = np.array(keras_model(x))

    import onnxruntime as ort
    ref = ort.InferenceSession(model.SerializeToString()).run(None, {'x': x})[0]
    assert np.allclose(out, ref, atol=1e-6)

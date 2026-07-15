import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx2kerastl import onnx_to_keras


def _build_model():
    # the torch GroupNorm-decomposition pattern: Reshape with a constant target containing
    # 0 ('keep input dim') where the kept dim is the dynamic batch
    shape_in = numpy_helper.from_array(np.array([0, 4, 64], dtype=np.int64), 'shape_in')
    shape_out = numpy_helper.from_array(np.array([0, 32, 8], dtype=np.int64), 'shape_out')
    nodes = [
        helper.make_node('Reshape', ['x', 'shape_in'], ['grouped']),
        helper.make_node('Relu', ['grouped'], ['activated']),
        helper.make_node('Reshape', ['activated', 'shape_out'], ['y']),
    ]
    graph = helper.make_graph(
        nodes, 'zero_dim_reshape',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 32, 8])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 32, 8])],
        [shape_in, shape_out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])


def test_zero_dim_reshape_with_dynamic_batch():
    model = _build_model()
    converted = onnx_to_keras(model, input_names=['x'], name_policy='attach_weights_name',
                              allow_partial_compilation=False)
    keras_model = converted.converted_model

    x = np.random.default_rng(0).standard_normal((3, 32, 8)).astype(np.float32)
    out = np.array(keras_model([x]))

    import onnxruntime as ort
    ref = ort.InferenceSession(model.SerializeToString()).run(None, {'x': x})[0]
    assert out.shape == ref.shape == (3, 32, 8)
    assert np.allclose(out, ref)

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx2kerastl import onnx_to_keras


def _build_model():
    # per-embodiment weight-bank selection (VLA action heads): 3D table gathered along
    # axis 0 by a per-sample runtime index, result used in a batched matmul
    table = numpy_helper.from_array(
        np.arange(8 * 6 * 4, dtype=np.float32).reshape(8, 6, 4), 'table')
    nodes = [
        helper.make_node('Gather', ['table', 'cat_ids'], ['bank'], axis=0),
        helper.make_node('MatMul', ['x', 'bank'], ['y']),
    ]
    graph = helper.make_graph(
        nodes, 'nd_gather',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 3, 6]),
         helper.make_tensor_value_info('cat_ids', TensorProto.INT64, ['batch'])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 3, 4])],
        [table])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])


def test_gather_3d_table_with_runtime_indices():
    model = _build_model()
    converted = onnx_to_keras(model, input_names=['x', 'cat_ids'],
                              name_policy='attach_weights_name',
                              allow_partial_compilation=False)
    keras_model = converted.converted_model

    x = np.random.default_rng(0).standard_normal((4, 3, 6)).astype(np.float32)
    cat_ids = np.array([0, 3, 7, 3], dtype=np.int64)
    out = np.array(keras_model([x, cat_ids]))

    import onnxruntime as ort
    ref = ort.InferenceSession(model.SerializeToString()).run(
        None, {'x': x, 'cat_ids': cat_ids})[0]
    assert out.shape == ref.shape == (4, 3, 4)
    assert np.allclose(out, ref)

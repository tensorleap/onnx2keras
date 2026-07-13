import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx2kerastl import onnx_to_keras


def _build_model():
    # y = table[t] + x : a per-sample scalar index input (t) next to a batched input (x),
    # the diffusion-timestep pattern
    table = numpy_helper.from_array(
        np.arange(40, dtype=np.float32).reshape(10, 4), 'table')
    gather = helper.make_node('Gather', ['table', 't'], ['emb'], axis=0)
    add = helper.make_node('Add', ['emb', 'x'], ['y'])
    graph = helper.make_graph(
        [gather, add], 'rank1_input',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 4]),
         helper.make_tensor_value_info('t', TensorProto.INT64, ['batch'])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 4])],
        [table])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])


def test_rank1_dynamic_batch_input_keeps_batch_dim():
    model = _build_model()
    converted = onnx_to_keras(model, input_names=['x', 't'], name_policy='attach_weights_name',
                              allow_partial_compilation=False)
    keras_model = converted.converted_model

    t_input = [t for t in keras_model.inputs if 't' in t.name.split('/')[-1]][0]
    assert tuple(t_input.shape) == (None,), f'expected (None,), got {t_input.shape}'

    x = np.ones((3, 4), dtype=np.float32)
    t = np.array([0, 2, 9], dtype=np.int64)
    out = np.array(keras_model([x, t]))

    import onnxruntime as ort
    ref = ort.InferenceSession(model.SerializeToString()).run(
        None, {'x': x, 't': t})[0]
    assert out.shape == ref.shape == (3, 4)
    assert np.allclose(out, ref), 'batched rows must not collapse to batch row 0'

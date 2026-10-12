import numpy as np
from onnx import helper, TensorProto
import onnxruntime as rt
import pytest

from onnx2kerastl import onnx_to_keras


def _model(axis, data_shape, indices):
    # Gather on two constants (the reference-point tables of a deformable-attention export):
    # both inputs are folded, so the converter gathers in numpy. Every axis must work.
    data = np.arange(int(np.prod(data_shape)), dtype=np.float32).reshape(data_shape)
    nodes = [
        helper.make_node("Constant", inputs=[], outputs=["data"],
                         value=helper.make_tensor("d", TensorProto.FLOAT, data_shape, data.ravel().tolist())),
        helper.make_node("Constant", inputs=[], outputs=["idx"],
                         value=helper.make_tensor("i", TensorProto.INT64, [len(indices)], indices)),
        helper.make_node("Gather", inputs=["data", "idx"], outputs=["g"], axis=axis),
        helper.make_node("Add", inputs=["g", "test_in"], outputs=["test_out"]),   # a live input keeps a graph
    ]
    out_shape = list(data_shape); out_shape[axis] = len(indices)
    graph = helper.make_graph(nodes=nodes, name="test-model",
                              inputs=[helper.make_tensor_value_info("test_in", TensorProto.FLOAT, out_shape)],
                              outputs=[helper.make_tensor_value_info("test_out", TensorProto.FLOAT, None)])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), out_shape


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_gather_on_constants_along_every_axis(axis):
    onnx_model, out_shape = _model(axis, (2, 3, 4, 5), [1, 0])
    np_input = np.zeros(out_shape, dtype=np.float32)
    sess = rt.InferenceSession(onnx_model.SerializeToString())
    onnx_out = sess.run(None, {"test_in": np_input})[0]
    keras_model = onnx_to_keras(onnx_model, ["test_in"], name_policy="attach_weights_name").converted_model
    keras_out = np.asarray(keras_model(np_input))
    assert keras_out.shape == onnx_out.shape
    np.testing.assert_allclose(keras_out, onnx_out)

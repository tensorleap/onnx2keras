import numpy as np
from onnx import helper, TensorProto
import onnxruntime as rt

from onnx2kerastl import onnx_to_keras


def _model(target):
    # torchvision's RPN: an (N, 1) objectness tensor reshaped to (num_images, -1) with num_images = 1.
    # The leading target dim is static and differs from the input's dim 0, so it is a real reshape,
    # not a keras.Reshape that keeps dim 0 as the batch.
    nodes = [
        helper.make_node("Constant", inputs=[], outputs=["shape"],
                         value=helper.make_tensor("shape_v", TensorProto.INT64, [len(target)], target)),
        helper.make_node("Reshape", inputs=["test_in", "shape"], outputs=["test_out"]),
    ]
    graph = helper.make_graph(
        nodes=nodes, name="test-model",
        inputs=[helper.make_tensor_value_info("test_in", TensorProto.FLOAT, [6, 1])],
        outputs=[helper.make_tensor_value_info("test_out", TensorProto.FLOAT, None)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_reshape_static_leading_dim_differs_from_input_dim0():
    onnx_model = _model([1, -1])
    np_input = np.arange(6, dtype=np.float32).reshape(6, 1)

    sess = rt.InferenceSession(onnx_model.SerializeToString())
    onnx_out = sess.run(None, {"test_in": np_input})[0]
    assert onnx_out.shape == (1, 6)

    keras_model = onnx_to_keras(onnx_model, ["test_in"], name_policy="attach_weights_name").converted_model
    assert tuple(keras_model.output.shape) == (1, 6)
    keras_out = np.asarray(keras_model(np_input))
    assert np.array_equal(keras_out, onnx_out)

import numpy as np
import onnx
import onnxruntime as rt
from onnx import TensorProto, helper, numpy_helper

from onnx2kerastl import onnx_to_keras


def _make_lstm_model(direction):
    rng = np.random.default_rng(42)
    seq_length = 4
    batch_size = 2
    input_size = 3
    hidden_size = 5
    num_directions = 2 if direction == "bidirectional" else 1

    w = rng.normal(size=(num_directions, 4 * hidden_size, input_size)).astype(np.float32)
    r = rng.normal(size=(num_directions, 4 * hidden_size, hidden_size)).astype(np.float32)
    b = rng.normal(size=(num_directions, 8 * hidden_size)).astype(np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "LSTM",
                inputs=["X", "W", "R", "B", "", "initial_h", "initial_c"],
                outputs=["Y", "Y_h", "Y_c"],
                hidden_size=hidden_size,
                direction=direction,
            )
        ],
        name=f"lstm_{direction}",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [seq_length, batch_size, input_size]),
            helper.make_tensor_value_info("initial_h", TensorProto.FLOAT, [num_directions, batch_size, hidden_size]),
            helper.make_tensor_value_info("initial_c", TensorProto.FLOAT, [num_directions, batch_size, hidden_size]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [seq_length, num_directions, batch_size, hidden_size]),
            helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [num_directions, batch_size, hidden_size]),
            helper.make_tensor_value_info("Y_c", TensorProto.FLOAT, [num_directions, batch_size, hidden_size]),
        ],
        initializer=[
            numpy_helper.from_array(w, name="W"),
            numpy_helper.from_array(r, name="R"),
            numpy_helper.from_array(b, name="B"),
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 8

    inputs = {
        "X": rng.normal(size=(seq_length, batch_size, input_size)).astype(np.float32),
        "initial_h": rng.normal(size=(num_directions, batch_size, hidden_size)).astype(np.float32),
        "initial_c": rng.normal(size=(num_directions, batch_size, hidden_size)).astype(np.float32),
    }
    return model, inputs


def _make_zero_copy_reshape_model():
    rng = np.random.default_rng(123)
    reshape_shape = np.array([0, 0, -1], dtype=np.int64)

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Constant", inputs=[], outputs=["reshape_shape"], value=numpy_helper.from_array(reshape_shape)),
            helper.make_node("Reshape", inputs=["X", "reshape_shape"], outputs=["Y"]),
        ],
        name="zero_copy_reshape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 5]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 20]),
        ],
    )

    reshaped_model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    reshaped_model.ir_version = 8
    inputs = {
        "X": rng.normal(size=(1, 3, 4, 5)).astype(np.float32),
    }
    return reshaped_model, inputs


def _run_onnx_and_keras(model, inputs):
    ort_session = rt.InferenceSession(model.SerializeToString())
    onnx_outputs = ort_session.run(None, inputs)

    input_names = list(inputs.keys())
    keras_model = onnx_to_keras(
        model,
        input_names,
        name_policy="attach_weights_name",
        allow_partial_compilation=False,
    ).converted_model
    keras_outputs = keras_model([inputs[name] for name in input_names])
    keras_outputs = [tensor.numpy() if hasattr(tensor, "numpy") else tensor for tensor in keras_outputs]
    return onnx_outputs, keras_outputs


def _assert_outputs_close(onnx_outputs, keras_outputs):
    for onnx_output, keras_output in zip(onnx_outputs, keras_outputs):
        np.testing.assert_allclose(keras_output, onnx_output, rtol=1e-4, atol=1e-4)


def test_forward_lstm_matches_onnxruntime():
    model, inputs = _make_lstm_model("forward")
    onnx_outputs, keras_outputs = _run_onnx_and_keras(model, inputs)
    _assert_outputs_close(onnx_outputs, keras_outputs)


def test_bidirectional_lstm_matches_onnxruntime():
    model, inputs = _make_lstm_model("bidirectional")
    onnx_outputs, keras_outputs = _run_onnx_and_keras(model, inputs)
    _assert_outputs_close(onnx_outputs, keras_outputs)


def test_zero_copy_reshape_matches_onnxruntime():
    model, inputs = _make_zero_copy_reshape_model()
    onnx_outputs, keras_outputs = _run_onnx_and_keras(model, inputs)
    np.testing.assert_allclose(keras_outputs[0], onnx_outputs[0][0], rtol=1e-4, atol=1e-4)

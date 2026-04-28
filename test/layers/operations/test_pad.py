"""Pad op: NHWC-style pads with many channels (c_last > 4) must use channels_last padding."""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from onnx2kerastl import onnx_to_keras


def _make_pad_model(pads: np.ndarray, input_shape, output_shape, opset=13):
    pads = np.asarray(pads, dtype=np.int64)
    x_info = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, list(input_shape)
    )
    y_info = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, list(output_shape)
    )
    pads_tensor = helper.make_tensor(
        "pads_const", TensorProto.INT64, [8], pads.tolist()
    )
    pad_node = helper.make_node(
        "Pad",
        inputs=["X", "pads_const"],
        outputs=["Y"],
        mode="constant",
    )
    graph = helper.make_graph(
        [pad_node],
        "pad_graph",
        [x_info],
        [y_info],
        initializer=[pads_tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model, full_check=True)
    return model


@pytest.mark.parametrize("channels", [3, 80])
def test_pad_nhwc_spatial_axes_high_channel_count(channels):
    """Pads axes 1,2 only (NHWC H/W); last axis is channels — must not use channels_first mapping."""
    pads = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
    input_shape = (1, 32, 32, channels)
    output_shape = (1, 34, 34, channels)
    onnx_model = _make_pad_model(pads, input_shape, output_shape)

    x = np.random.randn(*input_shape).astype(np.float32)
    ort_y = ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"X": x})[0]

    k = onnx_to_keras(
        onnx_model,
        ["X"],
        name_policy="short",
        allow_partial_compilation=False,
    ).converted_model
    keras_y = k(x, training=False).numpy()

    np.testing.assert_allclose(keras_y, ort_y, rtol=1e-5, atol=1e-5)


def test_pad_nchw_spatial_axes_unchanged():
    """Classic NCHW spatial pad: axes 2,3 — must keep channels_first mapping."""
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    input_shape = (1, 16, 32, 32)
    output_shape = (1, 16, 34, 34)
    onnx_model = _make_pad_model(pads, input_shape, output_shape)

    x = np.random.randn(*input_shape).astype(np.float32)
    ort_y = ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"X": x})[0]

    k = onnx_to_keras(
        onnx_model,
        ["X"],
        name_policy="short",
        allow_partial_compilation=False,
    ).converted_model
    keras_y = k(x, training=False).numpy()

    np.testing.assert_allclose(keras_y, ort_y, rtol=1e-5, atol=1e-5)

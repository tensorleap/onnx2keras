import numpy as np
import onnx
import onnxruntime as ort
import pytest

from onnx2kerastl import onnx_to_keras


@pytest.mark.parametrize(
    "onnx_path",
    ["test/models/private_tests/infineon/All_3ChipTypes_seg_model_deployed.onnx"],
)
def test_infineon_3chip_seg_model(onnx_path):
    onnx_model = onnx.load(onnx_path)

    ort_session = ort.InferenceSession(onnx_path)
    input_info = ort_session.get_inputs()[0]
    output_info = ort_session.get_outputs()[0]

    rng = np.random.default_rng(seed=42)
    input_np = rng.random((1, 250, 220, 1), dtype=np.float32)
    inputs = {input_info.name: input_np}

    onnx_out = ort_session.run([output_info.name], inputs)[0]

    keras_model = onnx_to_keras(
        onnx_model,
        [input_info.name],
        name_policy="attach_weights_name",
        allow_partial_compilation=False,
    ).converted_model

    keras_out = keras_model([input_np])
    if isinstance(keras_out, (list, tuple)):
        keras_out = keras_out[0]
    keras_np = keras_out.numpy() if hasattr(keras_out, "numpy") else keras_out

    diff = np.abs(keras_np - onnx_out)
    mean_error = diff.mean()
    max_error = diff.max()

    assert mean_error < 1e-4, f"mean error {mean_error:.6e} exceeds threshold"
    assert max_error < 1e-3, f"max error {max_error:.6e} exceeds threshold"

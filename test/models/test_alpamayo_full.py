import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from keras_data_format_converter import convert_channels_first_to_last

from onnx2kerastl import onnx_to_keras

# Full-model conversion test for nvidia/Alpamayo-R1-10B.
#
# The end-to-end VLA (images+text -> VLM generate loop -> 10-step diffusion loop -> trajectory)
# is Python control flow and does not trace to a single ONNX graph. The largest self-contained
# NEURAL forward is the diffusion denoiser step_fn, which runs the full 2.3B action expert
# transformer conditioned on the VLM KV-cache:
#     inputs : noisy_action (B, 64, 2), timesteps (B, 1, 1)
#     output : vector_field (B, 64, 2)
#
# Export that ONNX on a GPU box first (separate env - see test/models/alpamayo/export_alpamayo_full.py):
#     python test/models/alpamayo/export_alpamayo_full.py --out test/models/alpamayo/alpamayo_r1_full_denoiser.onnx
# then run this test in the onnx2kerastl env. Override the path with ALPAMAYO_FULL_ONNX_PATH.
DEFAULT_ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "alpamayo", "alpamayo_r1_full_denoiser.onnx"
)
ONNX_PATH = os.environ.get("ALPAMAYO_FULL_ONNX_PATH", DEFAULT_ONNX_PATH)

# The action expert is a deep transformer; fp32 conversion drift is larger than for a small MLP.
MEAN_TOL = float(os.environ.get("ALPAMAYO_FULL_MEAN_TOL", "1e-3"))
MAX_TOL = float(os.environ.get("ALPAMAYO_FULL_MAX_TOL", "1e-2"))

INT_DTYPES = {
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
}


def _normalize_shape(shape):
    normalized = []
    for dim in shape:
        if isinstance(dim, str) or dim is None:
            normalized.append(1)
        elif isinstance(dim, int) and dim > 0:
            normalized.append(dim)
        else:
            normalized.append(1)
    return tuple(normalized)


def _build_random_input(input_info, rng):
    shape = _normalize_shape(input_info.shape)

    if input_info.type == "tensor(bool)":
        return rng.integers(0, 2, size=shape).astype(bool)

    if input_info.type in INT_DTYPES:
        return rng.integers(0, 10, size=shape).astype(INT_DTYPES[input_info.type])

    return rng.random(shape).astype(np.float32)


def test_alpamayo_full_denoiser():
    if not os.path.exists(ONNX_PATH):
        pytest.skip(
            f"Alpamayo full-denoiser ONNX not found at {ONNX_PATH}. Export it on a GPU box with "
            f"test/models/alpamayo/export_alpamayo_full.py and set ALPAMAYO_FULL_ONNX_PATH."
        )

    onnx_model = onnx.load(ONNX_PATH)

    rng = np.random.default_rng(seed=42)
    session = ort.InferenceSession(ONNX_PATH)
    input_infos = session.get_inputs()
    output_infos = session.get_outputs()

    input_names = [info.name for info in input_infos]
    input_arrays = {info.name: _build_random_input(info, rng) for info in input_infos}
    output_names = [info.name for info in output_infos]

    # --------------------------------- Export to Keras -------------------------------------
    keras_model = onnx_to_keras(
        onnx_model,
        input_names,
        name_policy="attach_weights_name",
        allow_partial_compilation=False,
    ).converted_model
    final_model = convert_channels_first_to_last(
        keras_model, should_transform_inputs_and_outputs=False
    )

    # --------------------------------- Evaluating Inference -------------------------------------
    onnx_outputs = session.run(output_names, input_feed=input_arrays)
    keras_inputs = [input_arrays[name] for name in input_names]
    keras_outputs = final_model(keras_inputs)

    if not isinstance(keras_outputs, (list, tuple)):
        keras_outputs = [keras_outputs]

    assert len(keras_outputs) == len(onnx_outputs)

    for i, (keras_out, onnx_out) in enumerate(zip(keras_outputs, onnx_outputs)):
        keras_np = keras_out.numpy() if hasattr(keras_out, "numpy") else keras_out
        if onnx_out.dtype == bool or keras_np.dtype == bool:
            assert np.array_equal(keras_np, onnx_out), f"Output {i} boolean mismatch"
            continue

        if np.issubdtype(onnx_out.dtype, np.integer) and np.issubdtype(
            keras_np.dtype, np.integer
        ):
            assert np.array_equal(keras_np, onnx_out), f"Output {i} integer mismatch"
            continue

        diff = np.abs(keras_np - onnx_out)
        mean_error = diff.mean()
        max_error = diff.max()
        assert mean_error < MEAN_TOL, f"Output {i} mean error {mean_error} exceeds {MEAN_TOL}"
        assert max_error < MAX_TOL, f"Output {i} max error {max_error} exceeds {MAX_TOL}"


if __name__ == "__main__":
    test_alpamayo_full_denoiser()

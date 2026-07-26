import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from keras_data_format_converter import convert_channels_first_to_last

from onnx2kerastl import onnx_to_keras

# nvidia/Alpamayo-R1-10B is a custom Vision-Language-Action model (model_type `alpamayo_r1`,
# Cosmos-Reason / Qwen3-VL-8B backbone + 2.3B action expert + flow-matching diffusion decoder).
# It is NOT loadable through transformers AutoModel / optimum, and its inference is a VLM
# autoregressive rollout followed by a diffusion decode loop - which does not trace to a single
# ONNX graph. So we export a self-contained, faithfully-weighted piece of the diffusion denoiser:
# `action_in_proj` (PerWaypointActionInProjV2), which maps a noisy action sequence + diffusion
# timestep into expert-token embeddings (Fourier encoders + MLP + LayerNorm).
#
# The ONNX shipped here was exported from the real 10B weights (see scratchpad export script):
#   inputs : noisy_action (1, 64, 2)  [n_waypoints=64, accel/curvature], timesteps (1, 1, 1)
#   output : action_embeds (1, 64, 2048)
# Point the test elsewhere with ALPAMAYO_ONNX_PATH if you export a different submodule.
DEFAULT_ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "alpamayo", "alpamayo_r1_action_in_proj.onnx"
)
ONNX_PATH = os.environ.get("ALPAMAYO_ONNX_PATH", DEFAULT_ONNX_PATH)

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


def test_alpamayo_r1_10b():
    if not os.path.exists(ONNX_PATH):
        pytest.skip(
            f"Alpamayo ONNX not found at {ONNX_PATH}. Export it with NVIDIA's script "
            f"(transformers>=4.57.1, torch>=2.8) and set ALPAMAYO_ONNX_PATH."
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
        assert mean_error < 1e-4, f"Output {i} mean error {mean_error} exceeds threshold"
        assert max_error < 1e-3, f"Output {i} max error {max_error} exceeds threshold"


if __name__ == "__main__":
    test_alpamayo_r1_10b()

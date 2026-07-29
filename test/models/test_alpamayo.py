import gc
import os

import numpy as np
import onnx
import pytest
from keras_data_format_converter import convert_channels_first_to_last

from onnx2kerastl import onnx_to_keras

# Conversion tests for nvidia/Alpamayo-R1-10B, exported as 4 self-contained ONNX
# graphs that together cover every weight of the checkpoint (export scripts in
# test/models/alpamayo/, the .onnx artifacts are gitignored):
#   - embed_tokens:           input_ids -> inputs_embeds
#   - vision encoder (0.6B):  pixel patches -> visual embeds + deepstack embeds
#   - VLM decoder step (7.6B): inputs_embeds + past K/V x36 -> logits + present K/V x36
#   - action-expert denoiser (2.3B): noisy_action + timesteps + VLM KV-cache -> vector_field
#
# Generation loop = embed/vision -> decoder step per token, feeding present.*
# back as past_key_values.* (the past axis is dynamic); the denoiser then
# consumes the same cache tensors. test_alpamayo_decoder_kv_cache_loop
# simulates that rollout for a few steps.
#
# The decoder graph holds ~30GB of fp32 weights, so tests sequence memory
# explicitly: convert -> free ONNX proto -> run Keras -> free Keras -> run
# onnxruntime -> compare.
ALPAMAYO_DIR = os.environ.get(
    "ALPAMAYO_DIR", os.path.join(os.path.dirname(__file__), "alpamayo")
)

MEAN_TOL = float(os.environ.get("ALPAMAYO_MEAN_TOL", "1e-3"))
MAX_TOL = float(os.environ.get("ALPAMAYO_MAX_TOL", "1e-2"))

ONNX_TO_NP = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.BOOL: np.bool_,
}


def _model_path(onnx_name):
    path = os.path.join(ALPAMAYO_DIR, onnx_name)
    if not os.path.exists(path):
        pytest.skip(
            f"{onnx_name} not found under {ALPAMAYO_DIR}. Export it with the "
            f"scripts in test/models/alpamayo/ (see their docstrings) or set "
            f"ALPAMAYO_DIR."
        )
    return path


def _graph_input_specs(path):
    """(name, np_dtype, shape) per graph input, dynamic dims normalized to 1,
    read from the slim proto so the external weight data stays on disk."""
    model = onnx.load(path, load_external_data=False)
    specs = []
    for inp in model.graph.input:
        ttype = inp.type.tensor_type
        shape = tuple(
            d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1
            for d in ttype.shape.dim
        )
        specs.append((inp.name, ONNX_TO_NP[ttype.elem_type], shape))
    del model
    return specs


def _random_inputs(specs, rng):
    feeds = {}
    for name, dtype, shape in specs:
        if np.issubdtype(dtype, np.integer):
            feeds[name] = rng.integers(0, 10, size=shape).astype(dtype)
        elif dtype == np.bool_:
            feeds[name] = rng.integers(0, 2, size=shape).astype(bool)
        else:
            feeds[name] = rng.random(shape).astype(dtype)
    return feeds


def _load_keras(path, input_names):
    onnx_model = onnx.load(path)
    keras_model = onnx_to_keras(
        onnx_model,
        input_names,
        name_policy="attach_weights_name",
        allow_partial_compilation=False,
    ).converted_model
    del onnx_model
    gc.collect()
    # channels-last is the Tensorleap deployment format; also required to run
    # Conv3D (vision patch_embed) on CPU, where TF only implements NHWC
    keras_model = convert_channels_first_to_last(
        keras_model, should_transform_inputs_and_outputs=False
    )
    gc.collect()
    return keras_model


def _free_keras(keras_model):
    del keras_model
    try:
        from keras import backend as K

        K.clear_session()
    except Exception:
        pass
    gc.collect()


def _run_keras(keras_model, feeds, input_names):
    outputs = keras_model([feeds[name] for name in input_names])
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return [
        out.numpy() if hasattr(out, "numpy") else np.asarray(out) for out in outputs
    ]


def _assert_close(name, keras_out, onnx_out):
    assert keras_out.shape == onnx_out.shape, (
        f"{name}: shape {keras_out.shape} vs {onnx_out.shape}"
    )
    diff = np.abs(keras_out.astype(np.float64) - onnx_out.astype(np.float64))
    assert diff.mean() < MEAN_TOL, f"{name}: mean error {diff.mean()} > {MEAN_TOL}"
    assert diff.max() < MAX_TOL, f"{name}: max error {diff.max()} > {MAX_TOL}"


def _convert_and_compare(onnx_name):
    path = _model_path(onnx_name)
    specs = _graph_input_specs(path)
    feeds = _random_inputs(specs, np.random.default_rng(seed=42))
    input_names = [name for name, _, _ in specs]

    keras_model = _load_keras(path, input_names)
    keras_outputs = _run_keras(keras_model, feeds, input_names)
    _free_keras(keras_model)

    import onnxruntime as ort

    session = ort.InferenceSession(path)
    output_names = [info.name for info in session.get_outputs()]
    onnx_outputs = session.run(output_names, input_feed=feeds)
    del session
    gc.collect()

    assert len(keras_outputs) == len(onnx_outputs)
    for name, keras_out, onnx_out in zip(output_names, keras_outputs, onnx_outputs):
        _assert_close(name, keras_out, onnx_out)


def test_alpamayo_vlm_embed():
    _convert_and_compare("alpamayo_r1_vlm_embed.onnx")


def test_alpamayo_vision_encoder():
    _convert_and_compare("alpamayo_r1_vision_encoder.onnx")


def test_alpamayo_full_denoiser():
    _convert_and_compare("alpamayo_r1_full_denoiser.onnx")


# --------------------- KV-cache generation-loop simulation ---------------------
# The decoder step graph (7.6B) is only tested through the loop below — a
# single-pass compare would be a strict subset of it.
INIT_PAST_LEN = 4  # length of the fake "prompt" cache the loop starts from
N_GEN_STEPS = 3  # decode steps to simulate


def _decoder_step_inputs(path):
    """Split the decoder input specs into per-step tensors and KV-cache names."""
    specs = _graph_input_specs(path)
    input_names = [name for name, _, _ in specs]
    past_names = [n for n in input_names if n.startswith("past_key_values.")]
    kv_shape = next(shape for name, _, shape in specs if name in past_names)
    hidden = next(shape for name, _, shape in specs if name == "inputs_embeds")[-1]
    return input_names, past_names, kv_shape, hidden


def _decoder_loop_feeds(path, rng):
    """Pre-build the per-step feeds (everything except the evolving KV-cache)
    plus the initial cache, so the Keras and onnxruntime loops see identical
    inputs."""
    input_names, past_names, kv_shape, hidden = _decoder_step_inputs(path)
    batch, n_kv_heads, _, head_dim = kv_shape

    init_past = {
        name: (rng.standard_normal(
            (batch, n_kv_heads, INIT_PAST_LEN, head_dim)
        ) * 0.1).astype(np.float32)
        for name in past_names
    }

    steps = []
    for step in range(N_GEN_STEPS):
        past_len = INIT_PAST_LEN + step
        feeds = {
            "inputs_embeds": (rng.random((1, 1, hidden)) * 0.02).astype(np.float32),
            "position_ids": np.full((3, 1, 1), past_len, dtype=np.int64),
            "attention_mask": np.zeros((1, 1, 1, past_len + 1), dtype=np.float32),
        }
        for i in range(3):
            feeds[f"deepstack_embed_{i}"] = np.zeros((1, 1, hidden), dtype=np.float32)
        steps.append(feeds)
    return input_names, past_names, init_past, steps


def _run_decoder_loop(run_step, past_names, init_past, steps):
    """Run the decode loop: each step's present.* outputs become the next
    step's past_key_values.* inputs. run_step(feeds) -> [logits, *presents].
    Returns per-step logits and the final cache."""
    past = dict(init_past)
    all_logits = []
    for feeds in steps:
        outputs = run_step({**feeds, **past})
        all_logits.append(outputs[0])
        past = dict(zip(past_names, outputs[1:]))
    return all_logits, past


def test_alpamayo_decoder_kv_cache_loop():
    """Simulate a few autoregressive prediction steps through the converted
    decoder: save the KV-cache each step and feed it back for the next token,
    then check every step's logits and the final cache against onnxruntime
    running the same loop."""
    path = _model_path("alpamayo_r1_vlm_decoder.onnx")
    input_names, past_names, init_past, steps = _decoder_loop_feeds(
        path, np.random.default_rng(seed=7)
    )

    keras_model = _load_keras(path, input_names)
    keras_logits, keras_final_past = _run_decoder_loop(
        lambda feeds: _run_keras(keras_model, feeds, input_names),
        past_names, init_past, steps,
    )
    _free_keras(keras_model)

    import onnxruntime as ort

    session = ort.InferenceSession(path)
    output_names = [info.name for info in session.get_outputs()]
    onnx_logits, onnx_final_past = _run_decoder_loop(
        lambda feeds: session.run(output_names, input_feed=feeds),
        past_names, init_past, steps,
    )
    del session
    gc.collect()

    for step, (keras_out, onnx_out) in enumerate(zip(keras_logits, onnx_logits)):
        _assert_close(f"logits (step {step})", keras_out, onnx_out)
        assert keras_out.shape[:2] == (1, 1)
    for name in past_names:
        expected_len = INIT_PAST_LEN + N_GEN_STEPS
        assert keras_final_past[name].shape[2] == expected_len
        _assert_close(f"final {name}", keras_final_past[name], onnx_final_past[name])


if __name__ == "__main__":
    test_alpamayo_vlm_embed()
    test_alpamayo_vision_encoder()
    test_alpamayo_full_denoiser()
    test_alpamayo_decoder_kv_cache_loop()

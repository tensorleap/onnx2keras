import gc
import os

import numpy as np
import onnx
import pytest
from keras_data_format_converter import convert_channels_first_to_last

from onnx2kerastl import onnx_to_keras

# Conversion tests for the Alpamayo-R1-10B VLM backbone (Qwen3-VL-8B based),
# exported as three graphs with explicit KV-cache I/O by
# test/models/alpamayo/export_alpamayo_vlm.py:
#   - vision encoder (0.6B): pixel patches -> visual embeds + deepstack embeds
#   - decoder step (7.6B):   inputs_embeds + past K/V x36 -> logits + present K/V x36
#   - embed_tokens:          input_ids -> inputs_embeds
# Together with the expert denoiser (test_alpamayo_full.py) these cover every
# weight of the 10B checkpoint.
#
# The decoder graph holds ~30GB of fp32 weights, so these tests sequence memory
# explicitly: convert -> free ONNX proto -> run Keras -> free Keras -> run
# onnxruntime -> compare.
ALPAMAYO_DIR = os.environ.get(
    "ALPAMAYO_VLM_DIR", os.path.join(os.path.dirname(__file__), "alpamayo")
)

MEAN_TOL = float(os.environ.get("ALPAMAYO_VLM_MEAN_TOL", "1e-3"))
MAX_TOL = float(os.environ.get("ALPAMAYO_VLM_MAX_TOL", "1e-2"))

ONNX_TO_NP = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.BOOL: np.bool_,
}


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


def _convert_and_compare(onnx_name):
    path = os.path.join(ALPAMAYO_DIR, onnx_name)
    if not os.path.exists(path):
        pytest.skip(
            f"{onnx_name} not found under {ALPAMAYO_DIR}. Export it with "
            f"test/models/alpamayo/export_alpamayo_vlm.py (see its docstring) "
            f"or set ALPAMAYO_VLM_DIR."
        )

    specs = _graph_input_specs(path)
    rng = np.random.default_rng(seed=42)
    feeds = _random_inputs(specs, rng)
    input_names = [name for name, _, _ in specs]

    # ------------------------------- ONNX -> Keras -------------------------------
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

    keras_outputs = keras_model([feeds[name] for name in input_names])
    if not isinstance(keras_outputs, (list, tuple)):
        keras_outputs = [keras_outputs]
    keras_outputs = [
        out.numpy() if hasattr(out, "numpy") else np.asarray(out)
        for out in keras_outputs
    ]

    del keras_model
    try:
        from keras import backend as K

        K.clear_session()
    except Exception:
        pass
    gc.collect()

    # ------------------------------- onnxruntime ---------------------------------
    import onnxruntime as ort

    session = ort.InferenceSession(path)
    output_names = [info.name for info in session.get_outputs()]
    onnx_outputs = session.run(output_names, input_feed=feeds)
    del session
    gc.collect()

    assert len(keras_outputs) == len(onnx_outputs)
    for name, keras_out, onnx_out in zip(output_names, keras_outputs, onnx_outputs):
        assert keras_out.shape == onnx_out.shape, (
            f"{name}: shape {keras_out.shape} vs {onnx_out.shape}"
        )
        diff = np.abs(keras_out.astype(np.float64) - onnx_out.astype(np.float64))
        assert diff.mean() < MEAN_TOL, f"{name}: mean error {diff.mean()} > {MEAN_TOL}"
        assert diff.max() < MAX_TOL, f"{name}: max error {diff.max()} > {MAX_TOL}"


def test_alpamayo_vlm_embed():
    _convert_and_compare("alpamayo_r1_vlm_embed.onnx")


def test_alpamayo_vlm_vision_encoder():
    _convert_and_compare("alpamayo_r1_vision_encoder.onnx")


def test_alpamayo_vlm_decoder():
    _convert_and_compare("alpamayo_r1_vlm_decoder.onnx")


if __name__ == "__main__":
    test_alpamayo_vlm_embed()
    test_alpamayo_vlm_vision_encoder()
    test_alpamayo_vlm_decoder()

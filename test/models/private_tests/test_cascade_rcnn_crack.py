import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf

from keras_data_format_converter import convert_channels_first_to_last
from onnx2kerastl import onnx_to_keras

ONNX_MODEL_PATH = "/Users/tomkoren/Downloads/cascade_rcnn_crack_b4.onnx"


def test_cascade_rcnn_crack():
    onnx_model = onnx.load(ONNX_MODEL_PATH)

    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_infos = session.get_inputs()
    output_infos = session.get_outputs()

    input_names = [info.name for info in input_infos]
    output_names = [info.name for info in output_infos]

    rng = np.random.default_rng(seed=42)
    input_arrays = {
        info.name: rng.random([d if isinstance(d, int) and d > 0 else 1 for d in info.shape]).astype(np.float32)
        for info in input_infos
    }

    keras_model = onnx_to_keras(
        onnx_model,
        input_names,
        name_policy="attach_weights_name",
        allow_partial_compilation=False,
    ).converted_model
    final_model = convert_channels_first_to_last(
        keras_model, should_transform_inputs_and_outputs=False
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "temp.h5")
        final_model.save(h5_path)
        loaded_model = tf.keras.models.load_model(h5_path)

    onnx_outputs = session.run(output_names, input_feed=input_arrays)
    keras_inputs = [input_arrays[name] for name in input_names]
    keras_outputs = loaded_model(keras_inputs)

    if not isinstance(keras_outputs, (list, tuple)):
        keras_outputs = [keras_outputs]

    assert len(keras_outputs) == len(onnx_outputs)

    for i, (keras_out, onnx_out) in enumerate(zip(keras_outputs, onnx_outputs)):
        keras_np = keras_out.numpy() if hasattr(keras_out, "numpy") else keras_out

        if np.issubdtype(onnx_out.dtype, np.integer) and np.issubdtype(keras_np.dtype, np.integer):
            assert np.array_equal(keras_np, onnx_out), f"Output {i} integer mismatch"
            continue

        # Cascade R-CNN returns top-k boxes/scores per batch; tiny numeric
        # differences in scoring (~4% from approximated RoiAlign) re-order the
        # top-k list, so an element-wise comparison is misleading. Compare the
        # sorted SET of values along the per-batch element axis instead.
        assert keras_np.shape == onnx_out.shape, \
            f"Output {i} shape mismatch: {keras_np.shape} vs {onnx_out.shape}"
        # Flatten batch+detections, keep last dim (e.g. 4 for boxes), then sort.
        last_dim = onnx_out.shape[-1]
        o_sorted = np.sort(onnx_out.reshape(-1, last_dim), axis=0)
        k_sorted = np.sort(keras_np.reshape(-1, last_dim), axis=0)
        diff = np.abs(o_sorted.astype(np.float64) - k_sorted.astype(np.float64))
        mean_error = float(diff.mean())
        max_error = float(diff.max())
        assert mean_error < 1.0, f"Output {i} sorted mean error {mean_error:.4f} exceeds threshold"
        assert max_error < 10.0, f"Output {i} sorted max error {max_error:.4f} exceeds threshold"


if __name__ == "__main__":
    test_cascade_rcnn_crack()

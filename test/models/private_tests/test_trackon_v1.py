import random

import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
import onnxruntime as ort
from test.models.private_tests.aws_utils import aws_s3_download
import pytest


@pytest.mark.parametrize('aws_s3_download', [["asensus/", "asensus/", False]], indirect=True)
def test_trackon_v1_mode(aws_s3_download):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ort.set_seed(SEED)

    model_path = f"{aws_s3_download}/TrackerInferenceFcudarc3aug.onnx"
    onnx_model = onnx.load(model_path)
    keras_model = onnx_to_keras(onnx_model,
                                input_names=["f_proj", "mask", "q_init", "spatial_memory", "context_memory", "past_occ", "past_mask", "t", "vis_threshold"],
                                input_types=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.bool, tf.bool, tf.float32, tf.float32],
                                name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False, verbose=True)

    f_proj = np.random.random((1, 256, 128, 160)).astype(np.float32)
    mask = np.random.random((1, 512, 640)).astype(np.float32)
    q_init = np.random.random((1, 880, 256)).astype(np.float32)
    spatial_memory = np.random.random((1, 880, 48, 256)).astype(np.float32)
    context_memory = np.random.random((1, 880, 48, 256)).astype(np.float32)
    past_occ = np.ones((1, 880, 48), dtype=bool)
    past_mask = np.ones((1, 880, 48), dtype=bool)
    t = np.random.random((1, 1)).astype(np.float32)
    vis_threshold = np.random.random((1, 1)).astype(np.float32)

    keras_output = final_model([f_proj, mask, q_init, spatial_memory, context_memory, past_occ, past_mask, t, vis_threshold])

    ort_session = ort.InferenceSession(model_path)
    onnx_outputs = ort_session.run(None, {
        "f_proj": f_proj,
        "mask": mask,
        "q_init": q_init,
        "spatial_memory": spatial_memory,
        "context_memory": context_memory,
        "past_occ": past_occ,
        "past_mask": past_mask,
        "t": t,
        "vis_threshold": vis_threshold})

    for i, (onnx_out, keras_out) in enumerate(zip(onnx_outputs, keras_output)):
        keras_np = keras_out.numpy() if hasattr(keras_out, 'numpy') else keras_out

        if onnx_out.dtype == bool or keras_np.dtype == bool:
            mismatch = np.logical_xor(onnx_out, keras_np)
            num_mismatches = mismatch.sum()
            total_elements = mismatch.size
            assert num_mismatches == 0, f"Output {i}: {num_mismatches}/{total_elements} boolean mismatches"
        else:
            diff = np.abs(onnx_out - keras_np)
            mean_error = diff.mean()
            max_error = diff.max()
            assert max_error < 1e-3, f"Output {i}: max_error={max_error}, mean_error={mean_error}"


if __name__ == "__main__":
    test_trackon_v1_mode()

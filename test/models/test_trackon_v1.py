import random

import numpy as np
import onnx
from pyarrow import float32

from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
import onnxruntime as ort
import pytest
import pathlib



def test_trackon_v1_mode():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ort.set_seed(SEED)

    dir = pathlib.Path(__file__).parent.resolve()
    model_path = f"{dir}/TrackerInferenceFcudarc3aug.onnx"
    onnx_model = onnx.load(model_path)
    # convert onnx model to keras
    keras_model = onnx_to_keras(onnx_model,
                                input_names=["f_proj", "mask", "q_init", "spatial_memory", "context_memory", "past_occ", "past_mask", "t", "vis_threshold"],
                                # input_names=["context_memory", "t", "q_init", "past_mask", "past_occ", "spatial_memory",
                                #              "f_proj", "vis_threshold", "mask"],
                                input_types = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.bool, tf.bool, tf.float32, tf.float32],
                                name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    f_proj = np.random.random((1, 256, 128, 160)).astype(np.float32)
    mask = np.random.random((1, 512, 640)).astype(np.float32)
    q_init = np.random.random((1, 880, 256)).astype(np.float32)
    spatial_memory = np.random.random((1, 880, 48, 256)).astype(np.float32)
    context_memory = np.random.random((1, 880, 48, 256)).astype(np.float32)
    past_occ = np.ones((1, 880, 48), dtype=bool)
    past_mask = np.ones((1, 880, 48), dtype=bool)
    t = np.random.random((1, 1)).astype(np.float32)
    vis_threshold = np.random.random((1, 1)).astype(np.float32)

    #keras_output = final_model([f_proj, mask, q_init, spatial_memory, context_memory, past_occ, past_mask, t, vis_threshold])
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False, verbose=True)
    keras_output = final_model([context_memory, t, q_init, past_mask, past_occ, spatial_memory, f_proj, vis_threshold, mask])

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

    is_same = np.allclose(onnx_outputs, keras_output, 1e-6)

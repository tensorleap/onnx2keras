import numpy as np
import onnx
from pyarrow import float32

from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
import onnxruntime as ort
import pytest
import pathlib

def test_mnist_mode():
    # load onnx model
    dir = pathlib.Path(__file__).parent.resolve()
    model_path = f"{dir}/TrackerInferenceFcudarc3aug.onnx"
    onnx_model = onnx.load(model_path)
    # convert onnx model to keras
    keras_model = onnx_to_keras(onnx_model,
                                input_names=["f_proj", "mask", "q_init", "spatial_memory", "context_memory", "past_occ", "past_mask", "t", "vis_threshold"],
                                input_types = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.bool, tf.bool, tf.float32, tf.float32],
                                name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False, verbose=True)
    ort_session = ort.InferenceSession(model_path)
    f_proj = np.random.random((1, 880, 48, 256))
    mask = np.random.random((1, 512, 640))
    q_init = np.random.random((1, 880, 256))
    spatial_memory = np.random.random((1, 880, 48, 256))
    context_memory = np.random.random((1, 880, 48, 256))
    past_occ = np.zeros((1, 880, 48), dtype=bool) # np.random.random((1, 1, 28, 28))
    past_mask = np.zeros((1, 256, 128, 160), dtype=bool) # np.random.random((1, 1, 28, 28))
    t = np.random.random((1, 1))
    vis_threshold = np.random.random((1, 1))

    keras_output = final_model([f_proj, mask, q_init, spatial_memory, context_memory, past_occ, past_mask, t, vis_threshold])
    onnx_outputs = ort_session.run(None, {
        "f_proj": f_proj.astype(np.float32),
        "mask": mask.astype(np.float32),
        "q_init": q_init,
        "spatial_memory": spatial_memory,
        "context_memory": context_memory,
        "past_occ": past_occ,
        "past_mask": past_mask,
        "t": t.astype(np.float32),
        "vis_threshold": vis_threshold})

    is_same = np.allclose(onnx_outputs, keras_output, 1e-6)

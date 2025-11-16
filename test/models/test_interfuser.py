# code to proprely load data here: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
import onnx
import onnxruntime as ort

# from transformers.onnx import export, OnnxConfig
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
from packaging import version
from collections import OrderedDict
from typing import Mapping
import urllib


def test_interfuser():
    import torch
    model_name = 'interfuser'
    MODEL_PATH = './test/interfuser/interfuser_planKD_26_3M.onnx'
    onnx_model = onnx.load(MODEL_PATH)
    output_names = ['out0', 'out1', 'out2', 'out3', 'out4', 'out5']
    input_keys = ['lidar', 'measurements',  'rgb','rgb_center','rgb_left', 'rgb_right', 'target_point']
    shapes = [
        (1, 3, 224, 224),
        (1, 7),
        (1, 3, 256, 256),
        (1, 3, 128, 128),
        (1, 3, 256, 256),
        (1, 3, 256, 256),
        (1, 2)
    ]

    inputs = tuple(np.random.rand(*shape).astype(np.float32) for shape in shapes)
    inputs = {key: inp_ for key, inp_ in zip(input_keys, inputs)}
    keras_model = onnx_to_keras(onnx_model, input_keys, name_policy='attach_weights_name'
                                , allow_partial_compilation=False)
    keras_model = keras_model.converted_model
    inps_to_swap = ['bev', 'lidar', 'rgb', 'rgb2', 'rgb_center', 'rgb_left', 'rgb_right']
    inputs_swapped = {}
    for key in input_keys:
        if key in inps_to_swap:
            permuted_input = np.transpose(inputs[key], (0,2,3,1))
            inputs_swapped[key] = permuted_input
        else:
            inputs_swapped[key] = inputs[key]
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False)
    ort_session = ort.InferenceSession(MODEL_PATH)
    onnx_res = ort_session.run(
        output_names,
        input_feed=inputs)[0]
    keras_preds = final_model(inputs)[0]
    assert np.abs(keras_preds - onnx_res).max() < 1e-04

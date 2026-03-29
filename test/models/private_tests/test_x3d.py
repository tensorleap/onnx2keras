import onnx
import numpy as np
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import pytest
import onnxruntime as ort
import torch
from test.models.private_tests.aws_utils import aws_s3_download

@pytest.mark.parametrize('aws_s3_download', [["x3d/", "x3d/", False]], indirect=True)
def test_x3d(aws_s3_download):
    onnx_model_path = f'{aws_s3_download}/x3d.onnx'
    onnx_model = onnx.load(onnx_model_path)
    input_all = [_input.name for _input in onnx_model.graph.input]
    inputs = torch.rand(3,16,256,256)
    keras_model = onnx_to_keras(onnx_model, ['input'], name_policy='attach_weights_name', allow_partial_compilation=False)
    keras_model = keras_model.converted_model
    ort_session = ort.InferenceSession(onnx_model_path)
    onnx_pred = ort_session.run(['output'], input_feed={input_all[0]: np.expand_dims(inputs, 0).astype(np.float32)})[0]

    inputs_for_keras = np.transpose(inputs.numpy(), (1, 2, 3, 0))
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=True)
    keras_preds = final_model(inputs_for_keras[None, ...])

    assert np.abs(keras_preds - onnx_pred).max() < 1e-04

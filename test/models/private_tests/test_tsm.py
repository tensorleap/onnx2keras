import numpy as np
import onnx
import pytest
from keras_data_format_converter import convert_channels_first_to_last
from onnx2kerastl import onnx_to_keras
from test.utils import NP_SEED
import onnxruntime as ort
from test.models.private_tests.aws_utils import aws_s3_download


@pytest.mark.parametrize('aws_s3_download', [["tsm/", "tsm/", False]], indirect=True)
def test_tsm(aws_s3_download):
    np.random.seed(seed=NP_SEED)
    tsm_model_path = f'{aws_s3_download}/tsm.onnx'
    onnx_model = onnx.load(tsm_model_path)
    input_all = [_input.name for _input in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    k_model = onnx_to_keras(onnx_model, input_names, name_policy='attach_weights_name', allow_partial_compilation=False)
    flipped_model = convert_channels_first_to_last(k_model.converted_model, should_transform_inputs_and_outputs=False)
    input_np = np.random.uniform(0, 1, (1, 8, 3, 256, 256))
    keras_res = flipped_model(input_np).numpy()
    ort_session = ort.InferenceSession(tsm_model_path)
    onnx_res = ort_session.run(
        ['output'],
        input_feed={input_all[0]: input_np.astype(np.float32)})[0]
    d_res=np.abs(onnx_res-keras_res)
    assert  d_res.mean() < 5e-6
    assert  d_res.max() < 1e-5
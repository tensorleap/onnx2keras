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


def test_ctformer(
    onnx_path = "./test/ctformer/ctformer2.onnx",
) -> None:
    # if not onnx_path.exists():
    #     raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    onnx_model = onnx.load(onnx_path)

    rng = np.random.default_rng(seed=42)
    input_array = np.random.rand(1, 1, 64, 64).astype(np.float32)
    input_tensor = input_array

    keras_model = onnx_to_keras(onnx_model, input_names=['input'], name_policy='attach_weights_name'
                                , allow_partial_compilation=False)
    keras_model = keras_model.converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False)
    ort_session = ort.InferenceSession(onnx_path )
    onnx_res = ort_session.run(
        None,
        {"input": input_array},
    )[0]
    keras_preds = final_model(input_array)[0]
    assert np.abs(keras_preds - onnx_res).max() < 1e-04
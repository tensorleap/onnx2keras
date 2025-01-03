import onnxruntime as ort
import numpy as np
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last
import tensorflow as tf
import pytest


def ttt_amazonv2():
    model_path = f'/Users/tomkoren/Downloads/gengraspv2_4m_opset14.onnx'
    onnx_model = onnx.load(model_path)
    # keras_model = onnx_to_keras(onnx_model,
    #                             ['onnx::Transpose_0','onnx::Squeeze_1', 'onnx::Transpose_2', 'onnx::Gather_3'], name_policy='attach_weights_name',
    #                             allow_partial_compilation=False).converted_model
    keras_model = onnx_to_keras(onnx_model,
                                ['tensor.1', 'onnx::Slice_1'], name_policy='attach_weights_name',
                                allow_partial_compilation=False).converted_model
    final_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False)

    low_res = np.random.random((1, 1, 256, 256))
    full_res = np.random.random((1, 1, 1024, 2048))
    tuning_handles = np.random.random((1, 2, 1, 1))
    #cc([np.random.random((1, 256, 256, 1)), np.random.random((1, 1, 1024, 2048)), np.random.random((1, 1, 1, 2))])
    print(1)
   # res = final_model(np.transpose(img, [0, 2, 3, 1]))
    ort_session = ort.InferenceSession(model_path)
    #
    res_onnx = ort_session.run(
        ['1116'],
        input_feed={"low_res": low_res.astype(np.float32), 'full_res': full_res.astype(np.float32),
                    'tuning_handles': tuning_handles.astype(np.float32)}
    )[0]
    res = final_model([low_res, full_res, tuning_handles])
    final_model.save('a.h5')
    cc = tf.keras.models.load_model('a.h5')
    print(1)
    assert np.all(res_onnx[0]-res < 1e-5)
   #  eps_mean = 1e-6
   #  eps_max = 1e-5
   #  # These are really really close
   #
   #  assert (res[0] - res_onnx[0]).__abs__().numpy().mean() < 2 * eps_mean
   #  assert (res[0] - res_onnx[0]).__abs__().numpy().max() < 7 * eps_max
   #  assert (res[1] - res_onnx[1]).__abs__().numpy().mean() < eps_mean
   #  assert (res[1] - res_onnx[1]).__abs__().numpy().max() < eps_mean
   #
   #  assert (res[2] - res_onnx[2]).__abs__().numpy().mean() < eps_mean
   #  assert (res[2] - res_onnx[2]).__abs__().numpy().max() < eps_max
   #  assert (res[3] - res_onnx[3]).__abs__().numpy().mean() < eps_mean
   #  assert (res[3] - res_onnx[3]).__abs__().numpy().max() < eps_max
   #
   #  assert (res[4] - res_onnx[4]).__abs__().numpy().mean() < eps_mean
   #  assert (res[4] - res_onnx[4]).__abs__().numpy().max() < 7 * eps_max
   #
   #  assert (res[5][:, 0] - res_onnx[5]).__abs__().numpy().mean() < eps_mean
   #  assert (res[5][:, 0] - res_onnx[5]).__abs__().numpy().max() < eps_max
   #
   #  assert (res[6] - res_onnx[6]).__abs__().numpy().mean() < eps_mean
   #  assert (res[6] - res_onnx[6]).__abs__().numpy().max() < eps_max
   #
   #  # These two have lower accuracy but are still acceptable
   #
   #  (tf.nn.softmax(res[7][:4, :]) - tf.nn.softmax(res_onnx[7][:4, :])).numpy().__abs__().mean() < 5e-3
   #  (tf.nn.softmax(res[7][:4, :]) - tf.nn.softmax(res_onnx[7][:4, :])).numpy().__abs__().max() < 5e-2
   #
   #  (tf.nn.softmax(res[8][:4, :]) - tf.nn.softmax(res_onnx[8][:4, :])).numpy().__abs__().mean() < 5e-4
   #  (tf.nn.softmax(res[8][:4, :]) - tf.nn.softmax(res_onnx[8][:4, :])).numpy().__abs__().max() < 1e-2

ttt_amazonv2()
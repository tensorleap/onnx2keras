import numpy as np
import pytest
import tensorflow as tf
import onnx
from test.utils import convert_and_test
from test.detectron2_utils import get_detectron2_models_and_inputs
from onnx2keras import onnx_to_keras, check_torch_keras_error

@pytest.mark.slow
@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('model_class', ["COCO-Detection/retinanet_R_50_FPN_1x.yaml"])
def test_retinaNet(change_ordering, model_class):
    if change_ordering:
        pytest.skip("this doesn't work when you need to change ordering")
    # if not tf.test.gpu_device_name() and not change_ordering:
    #     pytest.skip("Skip! Since tensorflow Conv2D op currently only supports the NHWC tensor format on the CPU")
    # model = onnx.load(model_class)
    # model.eval()
    #
    # input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    onnx_model, model, inputs_dict = get_detectron2_models_and_inputs(model_class)
    k_model = onnx_to_keras(onnx_model, ["data", "im_info"], verbose=True, change_ordering=change_ordering)
    error = check_torch_keras_error(model, k_model, inputs_dict, change_ordering=change_ordering, epsilon=1e-5) #     2. 1x3 float "im_info", each row of which is (height, width, 1.0).

    print(1)
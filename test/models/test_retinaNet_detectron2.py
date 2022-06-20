run_test = False
if run_test: #note that to run this test you must use a different environment described in retinanet/RETINANET_README.md
    from onnx2kerastl import onnx_to_keras
    import torch
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.export import Caffe2Tracer
    import detectron2.data.transforms as T
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    import pytest
    import numpy as np
    from keras_data_format_converter import convert_channels_first_to_last
    from typing import Dict, List
    from torch import Tensor
    from detectron2.structures import ImageList
    from detectron2.modeling.meta_arch.dense_detector import DenseDetector
    from detectron2.export.shared import alias
    from detectron2.export.caffe2_modeling import Caffe2MetaArch


def preprocess_image(self, batched_inputs: List[Dict[str, Tensor]]):
    """
    Normalize, pad and batch the input images.
    """
    images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
    images = ImageList.from_tensors(
        images,
        self.backbone.size_divisibility,
        padding_constraints=self.backbone.padding_constraints,
    )
    return images


def get_sample_inputs(cfg, batch=False):
    # get a sample data
    original_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Do same preprocessing as DefaultPredictor
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    height, width = original_image.shape[:2]
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    inputs = {"image": image, "height": height, "width": width}
    if batch:
        second_image = image.detach().clone()
        second_input = {"image": second_image, "height": height, "width": width}
        sample_inputs = [inputs, second_input]
    else:
    # Sample ready
        sample_inputs = [inputs]
    return sample_inputs


def _caffe2_preprocess_image(self, inputs):
    """
    Caffe2 implementation of preprocess_image, which is called inside each MetaArch's forward.
    It normalizes the input images, and the final caffe2 graph assumes the
    inputs have been batched already.
    """
    data, im_info = inputs
    data = alias(data, "data")
    im_info = alias(im_info, "im_info")
    images = ImageList(tensor=data, image_sizes=im_info)
    return images


def get_detectron2_models_and_inputs(model_yaml_path:  str = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"):
    model_yaml = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
    cfg.MODEL.DEVICE = 'cpu'
    cfg['MODEL']['RETINANET']['TOPK_CANDIDATES_TEST'] = 10
    # This overloads the preprocess function to remove normalizing since switching channels fails on normalization
    DenseDetector.preprocess_image = preprocess_image
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    sample_inputs = get_sample_inputs(cfg=cfg, batch=False)
    Caffe2MetaArch._caffe2_preprocess_image = _caffe2_preprocess_image # remove normalization from preprocess
    tracer = Caffe2Tracer(cfg, model, sample_inputs) #this might raise w
    onnx_model = tracer.export_onnx() #this requires pip install onnx==1.8.1
    return onnx_model, model, sample_inputs


def check_torch_keras_error(model, k_model, input_dict, epsilon=1e-5, change_ordering=False,
                            should_transform_inputs=False):
    """
    Check difference between Torch and Keras models
    :param model: torch model
    :param k_model: keras model
    :param input_np: input data as numpy array or list of numpy array
    :param epsilon: allowed difference
    :param change_ordering: change ordering for keras input
    :param should_transform_inputs: default False, set to True for converting channel first inputs to  channel last format
    :return: actual difference

    """
    model.pixel_mean = torch.zeros_like(model.pixel_mean)
    images = model.preprocess_image(input_dict)
    features = model.backbone(images.tensor)
    features = [features[f] for f in model.head_in_features]
    pytorch_output = model.head(features)
    input_np = [np.expand_dims(np.pad(input_dict[0]['image'], [(0, 0), (0, 0), (0, 21)]), axis=0),
    np.expand_dims(np.array([input_dict[0]['height'], input_dict[0]['width'], 1.0]),
                    axis=0)]
    inputs_to_transpose = [k_input.name for k_input in k_model.inputs]
    _input_np = []
    for i in input_np:
        axes = list(range(len(i.shape)))
        axes = axes[0:1] + axes[2:] + axes[1:2]
        _input_np.append(np.transpose(i, axes))
    input_np = _input_np[::-1]
    k_model = convert_channels_first_to_last(k_model, inputs_to_transpose)
    keras_output = k_model(input_np)
    pytorch_output_flattened = [ele.detach().numpy() for i in range(2) for ele in pytorch_output[i]]
    pytorch_output_flattened = [np.swapaxes(np.swapaxes(ele, 1, 2), 2, 3) for ele in pytorch_output_flattened]
    keras_output_rearranged = keras_output[6::2] + keras_output[7::2]
    max_error = 0
    for p, k in zip(pytorch_output_flattened, keras_output_rearranged):
        error = np.max(np.abs(p - k))
        np.testing.assert_allclose(p, k, atol=epsilon, rtol=0.0)
        if error > max_error:
            max_error = error

    return max_error


@pytest.mark.slow
@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('model_class', ["COCO-Detection/retinanet_R_50_FPN_1x.yaml"])
def test_retinaNet(change_ordering, model_class):
    if change_ordering:
        pytest.skip("this doesn't work when you need to change ordering")
    if not change_ordering:
        if not run_test:
            pytest.skip("detectron2 needs a different env for testing")
    onnx_model, model, inputs_dict = get_detectron2_models_and_inputs(model_class)
    k_model = onnx_to_keras(onnx_model, ["data", "im_info"], verbose=True, change_ordering=change_ordering)
    error = check_torch_keras_error(model, k_model, inputs_dict, change_ordering=change_ordering, epsilon=1e-4) #     2. 1x3 float "im_info", each row of which is (height, width, 1.0)
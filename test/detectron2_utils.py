import torch
import pickle
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export import Caffe2Tracer
from detectron2.data import build_detection_test_loader, detection_utils
import detectron2.data.transforms as T
import pathlib
from os.path import join
import onnx
import pytest

def get_sample_inputs(sample_image, cfg, mode='test'):

    if sample_image is None:
        # get a first batch from dataset
        if mode != 'train':
            data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
            first_batch = next(iter(data_loader))
            return first_batch
        else:
            print("train")

    else:
        # get a sample data
        original_image = detection_utils.read_image(sample_image, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs


def get_detectron2_models_and_inputs(model_yaml_path:  str = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"):
    model = model_zoo.get(model_yaml_path, trained=True)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_yaml_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml_path)
    cfg.MODEL.DEVICE = 'cpu'
    sample_path = join(pathlib.Path(__file__).parent.resolve(), "data", "detectron2", "test.jpg")
    sample_inputs = get_sample_inputs(sample_image=sample_path, cfg=cfg)
    model.eval()
    model(sample_inputs)
    tracer = Caffe2Tracer(cfg, model, sample_inputs) #this might raise w
    onnx_model = tracer.export_onnx() #this requires pip install onnx==1.8.1
    return onnx_model, model, sample_inputs


import torch
import pickle
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export import Caffe2Tracer
from detectron2.data import build_detection_test_loader, detection_utils
import detectron2.data.transforms as T
import pathlib
from os.path import join
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import onnx
import pytest


def get_sample_inputs(sample_image, cfg, mode='test', batch=False):

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
        if batch:
            second_image = image.detach().clone()
            second_input = {"image": second_image, "height": height, "width": width}
            sample_inputs = [inputs, second_input]
        else:
        # Sample ready
            sample_inputs = [inputs]
        return sample_inputs


def get_detectron2_models_and_inputs(model_yaml_path:  str = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"):
    model_yaml = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
    cfg.MODEL.DEVICE = 'cpu'
    cfg['MODEL']['RETINANET']['TOPK_CANDIDATES_TEST'] = 10
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    sample_path = join(pathlib.Path(__file__).parent.resolve(), "data", "detectron2", "test.jpg")
    sample_inputs = get_sample_inputs(sample_image=sample_path, cfg=cfg, batch=True)
    model.eval()
    model(sample_inputs)
    # To consider this when building and mapping the priors
    classes = cfg['MODEL']['ROI_HEADS']['NUM_CLASSES']
    iou_thresholds = cfg['MODEL']['ROI_HEADS']['IOU_THRESHOLDS']
    nms_threshold = cfg['MODEL']['ROI_HEADS']['NMS_THRESH_TEST']
    lower_threshold = cfg['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST']
    loss = cfg['MODEL']['ROI_BOX_HEAD']['BBOX_REG_LOSS_TYPE']
    bbox_reg_weigts = cfg['MODEL']['ROI_BOX_HEAD']['BBOX_REG_WEIGHTS']
    smooth_l1_beta = cfg['MODEL']['ROI_BOX_HEAD']['SMOOTH_L1_BETA']
    bbox_cascade_reg = cfg['MODEL']['ROI_BOX_CASCADE_HEAD']['BBOX_REG_WEIGHTS']
    bbox_cascade_iou = cfg['MODEL']['ROI_BOX_CASCADE_HEAD']['IOUS']
    #
    tracer = Caffe2Tracer(cfg, model, sample_inputs) #this might raise w
    onnx_model = tracer.export_onnx() #this requires pip install onnx==1.8.1
    return onnx_model, model, sample_inputs


def get_detectron2_models_and_inputs_batch(model_yaml_path:  str = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"):
    model_yaml = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
    cfg.MODEL.DEVICE = 'cpu'
    cfg['MODEL']['RETINANET']['TOPK_CANDIDATES_TEST'] = 10
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    sample_path = join(pathlib.Path(__file__).parent.resolve(), "data", "detectron2", "test.jpg")
    sample_inputs = get_sample_inputs(sample_image=sample_path, cfg=cfg)
    model.eval()
    model(sample_inputs)
    # To consider this when building and mapping the priors
    classes = cfg['MODEL']['ROI_HEADS']['NUM_CLASSES']
    iou_thresholds = cfg['MODEL']['ROI_HEADS']['IOU_THRESHOLDS']
    nms_threshold = cfg['MODEL']['ROI_HEADS']['NMS_THRESH_TEST']
    lower_threshold = cfg['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST']
    loss = cfg['MODEL']['ROI_BOX_HEAD']['BBOX_REG_LOSS_TYPE']
    bbox_reg_weigts = cfg['MODEL']['ROI_BOX_HEAD']['BBOX_REG_WEIGHTS']
    smooth_l1_beta = cfg['MODEL']['ROI_BOX_HEAD']['SMOOTH_L1_BETA']
    bbox_cascade_reg = cfg['MODEL']['ROI_BOX_CASCADE_HEAD']['BBOX_REG_WEIGHTS']
    bbox_cascade_iou = cfg['MODEL']['ROI_BOX_CASCADE_HEAD']['IOUS']
    #
    tracer = Caffe2Tracer(cfg, model, sample_inputs) #this might raise w
    onnx_model = tracer.export_onnx() #this requires pip install onnx==1.8.1
    return onnx_model, model, sample_inputs

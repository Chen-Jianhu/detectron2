# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_yolov3_config(cfg):
    """
    Add config for YOLOV3.
    """
    # Darknet config
    cfg.MODEL.DARKNET = CN()
    cfg.MODEL.DARKNET.WEIGHTS = "https://pjreddie.com/media/files/darknet53_448.weights"
    cfg.MODEL.DARKNET.DEPTH = 53
    cfg.MODEL.DARKNET.OUT_FEATURES = ["stage3", "stage4", "stage5"]
    cfg.MODEL.DARKNET.STEM_OUT_CHANNELS = 32
    cfg.MODEL.DARKNET.NORM = "BN"

    # YOLOv3 config
    cfg.MODEL.YOLOV3 = CN()
    cfg.MODEL.YOLOV3.NUM_CLASSES = 80
    cfg.MODEL.YOLOV3.NORM = "BN"
    cfg.MODEL.YOLOV3.IN_FEATURES = ["stage3", "stage4", "stage5"]
    cfg.MODEL.YOLOV3.IOU_THRESHOLDS = [0.5, 0.5]
    cfg.MODEL.YOLOV3.IOU_LABELS = [0, -1, 1]
    cfg.MODEL.YOLOV3.BBOX_REG_WEIGHTS = [10.0, 10.0, 5.0, 5.0]
    # Loss parameters:
    cfg.MODEL.YOLOV3.LOSS_ALPHA = 1.0
    cfg.MODEL.YOLOV3.SMOOTH_L1_LOSS_BETA = 1.0
    cfg.MODEL.YOLOV3.NEGATIVE_POSITIVE_RATIO = 3.0
    # Inference parameters:
    cfg.MODEL.YOLOV3.SCORE_THRESH_TEST = 0.02
    cfg.MODEL.YOLOV3.NMS_THRESH_TEST = 0.45

    # Augmentations config
    cfg.INPUT.RESIZE = CN({"ENABLED": True})
    cfg.INPUT.RESIZE.SIZE = [320, 320]

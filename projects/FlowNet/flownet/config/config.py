# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/projects/FlowNet/flownet/config/config.py
@Time         : 2020-11-24 23:58:33
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-25 22:20:36
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

from detectron2.config import CfgNode as CN


def add_flownet_config(cfg):
    """
    Add config for FlowNet.
    """
    cfg.MODEL.FLOW_NET = CN()
    cfg.MODEL.FLOW_NET.NAME = "FlowNetS"
    cfg.MODEL.FLOW_NET.PIXEL_MEAN = [0.45, 0.432, 0.411]
    cfg.MODEL.FLOW_NET.PIXEL_STD = [1, 1, 1]
    cfg.MODEL.FLOW_NET.NEGATIVE_SLOPE = 0.1
    # training weight for each scale, from highest resolution (flow2) to lowest (flow6)
    cfg.MODEL.FLOW_NET.MULTISCALE_WEIGHTS = [0.005, 0.01, 0.02, 0.08, 0.32]
    # value by which flow will be divided.
    # Original value is 20 but 1 with batchNorm gives good results
    cfg.MODEL.FLOW_NET.FLOW_DIV = 20.

    # Data Augmentations
    cfg.INPUT.TRANSLATION = CN({"ENABLED": False})
    cfg.INPUT.TRANSLATION.RANGE = 0.2

    cfg.INPUT.ROTATION = CN({"ENABLED": False})
    cfg.INPUT.ROTATION.ANGLE = [-17, 17]

    cfg.INPUT.CROP = CN({"ENABLED": False})
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = [320, 448]

    cfg.INPUT.FLIP = ""

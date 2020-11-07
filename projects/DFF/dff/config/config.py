# -*- encoding: utf-8 -*-
"""
@File          :   config.py
@Time          :   2020/06/24 14:21:54
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:05:19
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from detectron2.config import CfgNode as CN


def add_dff_config(cfg):
    """
    Add config for DeepFeatureFlow.
    """
    # FlowNet config
    cfg.MODEL.FLOWNET = CN()
    cfg.MODEL.FLOWNET.NAME = "FlowNetS"
    cfg.MODEL.FLOWNET.WEIGHTS = ""
    cfg.MODEL.FLOWNET.PIXEL_MEAN = [0.45, 0.432, 0.411]
    cfg.MODEL.FLOWNET.PIXEL_STD = [1, 1, 1]
    cfg.MODEL.FLOWNET.NORM = ""
    cfg.MODEL.FLOWNET.NEGATIVE_SLOPE = 0.1
    # training weight for each scale, from highest resolution (flow2) to lowest (flow6)
    cfg.MODEL.FLOWNET.MULTISCALE_WEIGHTS = [0.005, 0.01, 0.02, 0.08, 0.32]
    # value by which flow will be divided.
    # Original value is 20 but 1 with batchNorm gives good results
    cfg.MODEL.FLOWNET.FLOW_DIV = 20.
    cfg.MODEL.FLOWNET.OUT_FEATURES = ["concat2", "flow2"]

    # DFF config
    cfg.MODEL.DFF = CN()
    cfg.MODEL.DFF.FRAME_OFFSET_RANGE = [-9, 0]
    cfg.MODEL.DFF.KEY_FRAME_DURATION = 10

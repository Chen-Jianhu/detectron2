# -*- encoding: utf-8 -*-
"""
@File          :   config.py
@Time          :   2020/06/20 7:16:21
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:23:55
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from detectron2.config import CfgNode as CN


def add_flownet_config(cfg):
    """
    Add config for FlowNet.
    """
    cfg.MODEL.FLOWNET = CN()
    cfg.MODEL.FLOWNET.NAME = "FlowNetS"
    cfg.MODEL.FLOWNET.PIXEL_MEAN = [0.45, 0.432, 0.411]
    cfg.MODEL.FLOWNET.PIXEL_STD = [1, 1, 1]
    cfg.MODEL.FLOWNET.NORM = "BN"
    cfg.MODEL.FLOWNET.NEGATIVE_SLOPE = 0.1
    # training weight for each scale, from highest resolution (flow2) to lowest (flow6)
    cfg.MODEL.FLOWNET.MULTISCALE_WEIGHTS = [0.005, 0.01, 0.02, 0.08, 0.32]
    # value by which flow will be divided.
    # Original value is 20 but 1 with batchNorm gives good results
    cfg.MODEL.FLOWNET.FLOW_DIV = 20.

    cfg.INPUT.TRANSFORMS = ()

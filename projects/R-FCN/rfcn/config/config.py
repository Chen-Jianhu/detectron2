# -*- encoding: utf-8 -*-
"""
@File          :   config.py
@Time          :   2020/06/23 19:22:06
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 22:59:44
@License       :   Copyright(C), USTC
@Desc          :   None
"""

# from detectron2.config import CfgNode as CN


def add_rfcn_config(cfg):
    """
    Add config for R-FCN.
    """
    cfg.MODEL.ROI_HEADS.NEW_CONV_CHANNELS = 1024

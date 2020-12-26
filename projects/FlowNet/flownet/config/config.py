# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/projects/FlowNet/flownet/config/config.py
@Time         : 2020-11-25 23:24:28
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-12-07 12:00:09
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

from detectron2.config import CfgNode as CN


def add_flownet_config(cfg):
    """
    Add config for FlowNet.
    """
    # Data Augmentations
    cfg.INPUT.TRANSLATION = CN({"ENABLED": False})
    cfg.INPUT.TRANSLATION.RANGE = 0.2

    cfg.INPUT.ROTATION = CN({"ENABLED": False})
    cfg.INPUT.ROTATION.ANGLE = [-17, 17]

    cfg.INPUT.CROP = CN({"ENABLED": False})
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = [320, 448]

    cfg.INPUT.FLIP = ""

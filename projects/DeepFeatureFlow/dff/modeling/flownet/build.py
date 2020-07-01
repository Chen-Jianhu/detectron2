# -*- encoding: utf-8 -*-
"""
@File          :   build.py
@Time          :   2020/06/27 19:06:47
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:07:07
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import torch

from detectron2.utils.registry import Registry

FLOWNET_REGISTRY = Registry("FLOWNET")
FLOWNET_REGISTRY.__doc__ = """
Registry for flownet, which preadict optical flow from image pairs.

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

It must returns an instance of :class:`Backbone`.
"""


def build_flownet(cfg):
    """
    Build a flownet from `cfg.MODEL.FLOWNET.NAME`.

    Returns:
        an instance of :class:`FlowNet`
    """
    flownet_name = cfg.MODEL.FLOWNET.NAME
    flownet = FLOWNET_REGISTRY.get(flownet_name)(cfg)
    flownet.to(torch.device(cfg.MODEL.DEVICE))
    return flownet

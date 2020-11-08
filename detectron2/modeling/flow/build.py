# -*- encoding: utf-8 -*-
"""
@File          :   build.py
@Time          :   2020/06/24 1:02:38
@Author        :   Chen-Jianhu (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:28:19
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import torch

from detectron2.utils.registry import Registry

FLOW_NET_REGISTRY = Registry("FLOW_NET")
FLOW_NET_REGISTRY.__doc__ = """
Registry for flow net, which preadict optical flow from image pairs.

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

It must returns an instance of :class:`Backbone`.
"""


def build_flow_net(cfg):
    """
    Build a flow net from `cfg.MODEL.FLOW_NET.NAME`.

    Returns:
        an instance of :class:`FLOW_NET`
    """
    flow_net_name = cfg.MODEL.FLOW_NET.NAME
    flow_net = FLOW_NET_REGISTRY.get(flow_net_name)(cfg)
    flow_net.to(torch.device(cfg.MODEL.DEVICE))
    return flow_net

# -*- encoding: utf-8 -*-
"""
@File          :   __init__.py
@Time          :   2020/06/27 19:06:47
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:07:02
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from .build import FLOWNET_REGISTRY, build_flownet

from .flownets import FlowNetS

__all__ = [
    "FLOWNET_REGISTRY",
    "build_flownet",
    "FlowNetS",
]

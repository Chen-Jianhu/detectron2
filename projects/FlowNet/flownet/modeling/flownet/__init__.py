# -*- encoding: utf-8 -*-
"""
@File          :   __init__.py
@Time          :   2020/06/20 7:47:36
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:28:12
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from .build import FLOWNET_REGISTRY, build_flownet

from .flownets import FlowNetS
from .flownetc import FlowNetC

__all__ = [
    "FLOWNET_REGISTRY",
    "build_flownet",
    "FlowNetS",
    "FlowNetC",
]

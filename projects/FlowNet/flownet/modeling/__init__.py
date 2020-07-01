# -*- encoding: utf-8 -*-
"""
@File          :   __init__.py
@Time          :   2020/06/20 7:48:50
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:29:03
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from .flownet import FlowNetS, FlowNetC

__all__ = [
    "FlowNetS",
    "FlowNetC",
]

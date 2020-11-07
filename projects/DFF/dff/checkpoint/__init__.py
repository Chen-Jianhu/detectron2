# -*- encoding: utf-8 -*-
"""
@File          :   __init__.py
@Time          :   2020/06/28 17:06:25
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:04:55
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from .detection_checkpoint import DetectionCheckpointer

__all__ = [
    "DetectionCheckpointer"
]

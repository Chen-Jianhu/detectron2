# -*- encoding: utf-8 -*-
"""
@File          :   __init__.py
@Time          :   2020/06/22 23:57:29
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 22:59:50
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from .roi_heads import RFCNROIHeads

__all__ = [
    "RFCNROIHeads",
]

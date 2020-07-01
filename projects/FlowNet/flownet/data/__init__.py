# -*- encoding: utf-8 -*-
"""
@File          :   __init__.py
@Time          :   2020/06/20 7:40:51
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:25:39
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from .dataset_mapper import FlyingChairsMapper
# ensure the builtin datasets are registered
from . import datasets

__all__ = [
    "FlyingChairsMapper",
    "datasets",
]

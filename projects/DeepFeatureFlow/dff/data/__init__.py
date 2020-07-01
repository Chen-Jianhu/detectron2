# -*- encoding: utf-8 -*-
"""
@File          :   __init__.py
@Time          :   2020/06/22 12:12:16
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:06:15
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from .dataset_mapper import DatasetMapper
from . import datasets  # ensure the builtin datasets are registered

__all__ = [
    "DatasetMapper",
    "datasets",
]

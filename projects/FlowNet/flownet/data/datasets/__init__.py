# -*- encoding: utf-8 -*-
"""
@File          :   __init__.py
@Time          :   2020/06/20 7:10:32
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:24:48
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from . import builtin  # ensure the builtin datasets are registered


__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]

# -*- encoding: utf-8 -*-
"""
@File          :   builtin.py
@Time          :   2020/06/20 7:10:46
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:25:17
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import os

from .flying_chairs import get_flying_chairs_meta, register_flying_chairs

# ==== Predefined datasets and splits for FlyingChairs ==========


_PREDEFINED_SPLITS_FLYING_CHAIRS = {
    "flying_chairs_train": "FlyingChairs_release/flying_chairs_train.json",
    "flying_chairs_test": "FlyingChairs_release/flying_chairs_test.json",
}


def register_all_flying_chairs(root):
    meta = get_flying_chairs_meta()
    for (name, json_file) in _PREDEFINED_SPLITS_FLYING_CHAIRS.items():
        register_flying_chairs(
            name,
            meta,
            os.path.join(root, json_file)
        )


# Register it under detectron2/project/FlowNet/
_root = os.path.abspath(__file__).split("FlowNet")[0]
_root = os.path.join(_root, "FlowNet", "datasets")
register_all_flying_chairs(_root)

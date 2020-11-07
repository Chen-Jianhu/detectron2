# -*- encoding: utf-8 -*-
"""
@File          :   builtin.py
@Time          :   2020/06/22 12:13:12
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:05:36
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import os

from .imagenet_vid import get_imagenet_vid_instances_meta, register_imagenet_vid_instances


# ==== Predefined datasets and splits for IMAGENET ==========


_PREDEFINED_SPLITS_IMAGENET = {
    "imagenet_det_train_30classes": (
        "ILSVRC2015/Data/DET",
        "ILSVRC2015/Annotations/DET",
        "ILSVRC2015/ImageSets/DET_train_30classes.txt",
    ),
    "imagenet_vid_train_15frames": (
        "ILSVRC2015/Data/VID",
        "ILSVRC2015/Annotations/VID",
        "ILSVRC2015/ImageSets/VID_train_15frames.txt",
    ),
    "imagenet_vid_train_every10frames": (
        "ILSVRC2015/Data/VID",
        "ILSVRC2015/Annotations/VID",
        "ILSVRC2015/ImageSets/VID_train_every10frames.txt",
    ),
    "imagenet_vid_val_frames": (
        "ILSVRC2015/Data/VID",
        "ILSVRC2015/Annotations/VID",
        "ILSVRC2015/ImageSets/VID_val_frames.txt",
    ),
    "imagenet_vid_val_videos": (
        "ILSVRC2015/Data/VID",
        "ILSVRC2015/Annotations/VID",
        "ILSVRC2015/ImageSets/VID_val_videos.txt",
    ),
}


def register_all_imagenet_vid(root):
    meta = get_imagenet_vid_instances_meta()
    for (name, (img_dir, anno_dir, img_index)) in _PREDEFINED_SPLITS_IMAGENET.items():
        register_imagenet_vid_instances(
            name,
            meta,
            os.path.join(root, img_dir),
            os.path.join(root, anno_dir),
            os.path.join(root, img_index),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_imagenet_vid(_root)

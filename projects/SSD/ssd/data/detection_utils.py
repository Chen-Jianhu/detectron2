# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/projects/SSD/ssd/data/detection_utils.py
@Time         : 2020-11-24 17:43:21
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-24 23:34:47
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

from detectron2.data import transforms as T

from .transforms import (
    ColorAugTransform,
    RandomSwapChannelsTransform,
    Expand,
    MinIoURandomCrop
)


def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    augmentation = []

    if is_train:
        if cfg.INPUT.COLOR_AUG:
            augmentation.append(
                ColorAugTransform(img_format=cfg.INPUT.FORMAT)
            )

        if cfg.INPUT.SWAP_CHANNELS:
            augmentation.append(
                RandomSwapChannelsTransform()
            )

        if cfg.INPUT.EXPAND:
            augmentation.append(
                Expand(img_value=cfg.MODEL.PIXEL_MEAN)
            )

        if cfg.INPUT.MIN_IOU_CROP:
            augmentation.append(
                MinIoURandomCrop()
            )

    if cfg.INPUT.RESIZE.ENABLED:
        shape = cfg.INPUT.RESIZE.SIZE
        augmentation.append(T.Resize(shape))

    if is_train:
        augmentation.append(
            T.RandomFlip()
        )

    return augmentation

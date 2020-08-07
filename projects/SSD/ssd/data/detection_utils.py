# -*- coding: utf-8 -*-

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

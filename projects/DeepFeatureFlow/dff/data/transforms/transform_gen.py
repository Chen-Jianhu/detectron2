# -*- encoding: utf-8 -*-
"""
@File          :   transform_gen.py
@Time          :   2020/06/29 14:09:39
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:06:03
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from detectron2.data.transforms.transform_gen import check_dtype, TransformGen
from detectron2.data.transforms import Transform, TransformList


def apply_transform_gens(transform_gens, imgs):
    """
    Apply a list of :class:`TransformGen` or :class:`Transform` on the input image, and
    returns the transformed image and a list of transforms.

    We cannot simply create and return all transforms without
    applying it to the image, because a subsequent transform may
    need the output of the previous one.

    Args:
        transform_gens (list): list of :class:`TransformGen` or :class:`Transform` instance to
            be applied.
        imgs (list[ndarray] or ndarray): uint8 or floating point images with 1 or 3 channels.

    Returns:
        list[ndarray]: the transformed image
        TransformList: contain the transforms that's used.
    """
    for g in transform_gens:
        assert isinstance(g, (Transform, TransformGen)), g

    if not isinstance(imgs, list):
        imgs = [imgs]

    # Shape must be equal
    shape = imgs[0].shape
    for img in imgs:
        assert shape == img.shape
        check_dtype(img)

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(imgs[0]) if isinstance(g, TransformGen) else g
        assert isinstance(
            tfm, Transform
        ), "TransformGen {} must return an instance of Transform! Got {} instead".format(g, tfm)

        for idx, img in enumerate(imgs):
            imgs[idx] = tfm.apply_image(img)

        tfms.append(tfm)

    return imgs, TransformList(tfms)

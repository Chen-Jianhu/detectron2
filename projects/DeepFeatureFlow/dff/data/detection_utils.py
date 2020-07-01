# -*- encoding: utf-8 -*-
"""
@File          :   detection_utils.py
@Time          :   2020/06/29 22:58:48
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:06:30
@License       :   Copyright(C), USTC
@Desc          :   None
"""

from PIL import Image

from fvcore.common.file_io import PathManager

from detectron2.data.detection_utils import convert_PIL_to_numpy


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray): an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        # try:
        #     image = ImageOps.exif_transpose(image)
        # except Exception:
        #     pass

        return convert_PIL_to_numpy(image, format)

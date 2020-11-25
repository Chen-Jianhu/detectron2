# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/projects/FlowNet/flownet/data/dataset_mapper.py
@Time         : 2020-11-24 23:58:33
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-25 22:19:55
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

import copy
import logging
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils, flow_utils
from detectron2.data import transforms as T
from . import transforms_flow as TF

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    augmentation = []

    if is_train:
        if cfg.INPUT.TRANSLATION.ENABLED:
            offset_range = cfg.INPUT.TRANSLATION.RANGE
            augmentation.append(TF.RandomTranslation(offset_range))
        if cfg.INPUT.ROTATION.ENABLED:
            angle = cfg.INPUT.ROTATION.ANGLE
            augmentation.append(TF.RandomRotation(angle))
        if cfg.INPUT.CROP.ENABLED:
            crop_type = cfg.INPUT.CROP.TYPE
            crop_size = cfg.INPUT.CROP.SIZE
            augmentation.append(TF.RandomCrop(crop_type, crop_size))
        if cfg.INPUT.FLIP != "":
            augmentation.append(TF.RandomFlip(cfg.INPUT.FLIP))

    return augmentation


def _check_shape(img1: np.ndarray, img2: np.ndarray, flow: np.ndarray):
    assert img1.shape[:2] == img2.shape[:2] == flow.shape[:2], (
        "Different shape between image and flow: img1({}), img2({}), flow({})".format(
            img1.shape[:2], img2.shape[:2], flow.shape[:2]
        )
    )


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # image1 = utils.convert_PIL_to_numpy(
        #     Image.open(dataset_dict["image_file1"]), format=self.image_format)
        # image2 = utils.convert_PIL_to_numpy(
        #     Image.open(dataset_dict["image_file2"]), format=self.image_format)
        image1 = utils.read_image(dataset_dict["image_file1"], format=self.image_format)
        image2 = utils.read_image(dataset_dict["image_file2"], format=self.image_format)
        flow_map = flow_utils.read_flow(dataset_dict["flow_map_file"])
        _check_shape(image1, image2, flow_map)

        height, width = image1.shape[:2]  # h, w
        dataset_dict["height"] = height
        dataset_dict["width"] = width

        # Apply augmentations
        aug_input = T.AugInput(image=image1)
        transforms = self.augmentations(aug_input)
        image1 = aug_input.image
        image2 = transforms.apply_image2(image2)
        flow_map = transforms.apply_flow(flow_map)
        _check_shape(image1, image2, flow_map)

        # Visualize
        # from detectron2.utils.flow_visualizer import (
        #     visualize_sample_from_array,
        #     visualize_sample_from_file
        # )
        # visualize_sample_from_array(image1, image2, flow_map, save=True)
        # visualize_sample_from_file(
        #     dataset_dict["image_file1"],
        #     dataset_dict["image_file2"],
        #     dataset_dict["flow_map_file"],
        #     save=True
        # )

        dataset_dict["image1"] = torch.as_tensor(
            np.ascontiguousarray(image1.transpose(2, 0, 1)))
        dataset_dict["image2"] = torch.as_tensor(
            np.ascontiguousarray(image2.transpose(2, 0, 1)))
        dataset_dict["flow_map"] = torch.as_tensor(
            np.ascontiguousarray(flow_map.transpose(2, 0, 1)))

        return dataset_dict

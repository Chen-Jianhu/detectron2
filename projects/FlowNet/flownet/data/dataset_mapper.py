# -*- encoding: utf-8 -*-
"""
@File          :   dataset_mapper.py
@Time          :   2020/06/20 7:17:07
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:25:49
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import copy
import logging
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data.flow_utils import read_flow

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["FlyingChairsMapper"]


logger = logging.getLogger(__name__)


class FlyingChairsMapper:
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

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.TRANSFORMS and is_train:
            # self.tfms = build_transforms(cfg.INPUT.TRANSFORMS)
            logger.info("Transform used in training: " + str(self.tfms))
        else:
            self.tfms = None

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # USER: Write your own image loading if it's not from a file
        # image1 = utils.convert_PIL_to_numpy(
        #     Image.open(dataset_dict["image_file1"]), format=self.img_format)
        # image2 = utils.convert_PIL_to_numpy(
        #     Image.open(dataset_dict["image_file2"]), format=self.img_format)
        image1 = utils.read_image(dataset_dict["image_file1"], format=self.img_format)
        image2 = utils.read_image(dataset_dict["image_file2"], format=self.img_format)
        assert image1.shape == image2.shape

        height, width = image1.shape[:2]  # h, w
        dataset_dict["height"] = height
        dataset_dict["width"] = width

        flow_map = read_flow(dataset_dict["flow_map_file"])

        # Apply transforms
        if self.tfms:
            images = (image1, image2)
            images, flow_map = self.tfms(images, flow_map)
            (image1, image2) = images

        # from .utils import visualize_sample_from_array, visualize_sample_from_file
        # visualize_sample_from_array(image1, image2, flow_map)
        # visualize_sample_from_file(
        #     dataset_dict["image_file1"],
        #     dataset_dict["image_file2"],
        #     dataset_dict["flow_map_file"],
        # )

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image1"] = torch.as_tensor(
            np.ascontiguousarray(image1.transpose(2, 0, 1)))
        dataset_dict["image2"] = torch.as_tensor(
            np.ascontiguousarray(image2.transpose(2, 0, 1)))
        dataset_dict["flow_map"] = torch.as_tensor(
            np.ascontiguousarray(flow_map.transpose(2, 0, 1)))

        return dataset_dict

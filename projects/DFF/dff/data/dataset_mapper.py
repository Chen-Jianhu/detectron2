# -*- encoding: utf-8 -*-
"""
@File          :   dataset_mapper.py
@Time          :   2020/06/22 12:12:16
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:06:22
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import logging
import os
import copy
import numpy as np
import torch
import random

from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog

from .transforms import apply_transform_gens
from .detection_utils import read_image

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]

logger = logging.getLogger(__name__)


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

    def __init__(self, cfg, is_train=True):
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.frame_offset_min = cfg.MODEL.DFF.FRAME_OFFSET_RANGE[0]
        self.frame_offset_max = cfg.MODEL.DFF.FRAME_OFFSET_RANGE[1]
        self.key_frame_duration = cfg.MODEL.DFF.KEY_FRAME_DURATION
        # fmt: on

        self.is_train = is_train
        # Same over all VID dataset
        self.vid_image_root = MetadataCatalog.get("imagenet_vid_train_15frames").img_dir
        # For testing
        self.key_frame_img = None
        self.key_frame_seg_id = None

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # USER: Write your own image loading if it's not from a file
        image_cur = read_image(dataset_dict["file_name"], format=self.img_format)

        # =============== Train ===============
        if self.is_train:
            pattern = dataset_dict.pop("pattern", None)
            if pattern is None:
                # DET set
                image_ref = image_cur.copy()
            else:
                # VID set
                offset = random.randint(self.frame_offset_min, self.frame_offset_max + 1)
                ref_id = min(
                    max(dataset_dict["frame_seg_id"] + offset, 0),
                    dataset_dict["frame_seg_len"] - 1
                )
                ref_file_name = os.path.join(self.vid_image_root, pattern.format(ref_id))
                image_ref = read_image(ref_file_name, format=self.img_format)

        # =============== Test ===============
        else:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)

            frame_seg_id = dataset_dict["frame_seg_id"]
            if frame_seg_id % self.key_frame_duration == 0:
                image_ref = image_cur.copy()
                self.key_frame_img = image_cur.copy()
                self.key_frame_seg_id = frame_seg_id
            else:
                try:
                    assert self.key_frame_img is not None
                    assert 0 < frame_seg_id - self.key_frame_seg_id < self.key_frame_duration, (
                        "frame_seg_id is {}, key_frame_seg_id is {}.".format(
                            frame_seg_id, self.key_frame_seg_id
                        )
                    )
                except AssertionError as e:
                    logger.warning(e)
                    logger.warning(dataset_dict)

                image_ref = self.key_frame_img

            frame_seg_len = dataset_dict["frame_seg_len"]
            if (frame_seg_id + 1) == frame_seg_len:
                self.key_frame_img = None
                self.key_frame_seg_id = None

        utils.check_image_size(dataset_dict, image_cur)
        utils.check_image_size(dataset_dict, image_ref)

        (image_cur, image_ref), transforms = apply_transform_gens(
            self.tfm_gens, [image_cur, image_ref]
        )

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image_cur"] = torch.as_tensor(
            np.ascontiguousarray(image_cur.transpose(2, 0, 1)))
        dataset_dict["image_ref"] = torch.as_tensor(
            np.ascontiguousarray(image_ref.transpose(2, 0, 1)))

        if "annotations" in dataset_dict:
            image_shape = image_cur.shape[:2]  # h, w
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)

            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

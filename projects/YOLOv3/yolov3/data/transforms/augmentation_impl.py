# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import numpy as np

from fvcore.transforms.transform import NoOpTransform

from detectron2.data.transforms import Augmentation
from detectron2.structures import pairwise_iou, Boxes

from .transform import ExpandTransform, CropTransform


__all__ = [
    "Expand",
    "MinIoURandomCrop",
]


class Expand(Augmentation):
    """
    Random Expand the image & bboxes.
    """

    def __init__(self, ratio_range=(1.0, 4.0), img_value=0, seg_value=255, prob=0.5):
        """
        Args:
            ratio_range (tuple): range of expand ratio.
            img_value (tuple): mean value of dataset.
            seg_value (float): probability of applying this transformation.
        """
        super().__init__()
        self._init(locals())
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"

    def get_transform(self, img):
        do = self._rand_range() < self.prob
        if do:
            h, w, _ = img.shape
            ratio = np.random.uniform(*self.ratio_range)
            top = int(np.random.uniform(0, h * ratio - h))
            left = int(np.random.uniform(0, w * ratio - w))
            return ExpandTransform(ratio, top, left, self.img_value, self.seg_value)
        else:
            return NoOpTransform()


class MinIoURandomCrop(Augmentation):
    """
    Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.
    """

    input_args = ("image", "boxes")

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        """
        Args:
            min_ious (tuple): minimum IoU threshold for all intersections with bounding boxes
            min_crop_size (float): minimum crop's size
                (i.e. h,w := a*h, a*w, where a >= min_crop_size).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, boxes):
        """
        Args:
            img (ndarray): of shape HxWxC(RGB). The array can be of type uint8
                in range [0, 255], or floating point in range [0, 255].
            annotations (list[dict[str->str]]):
                Each item in the list is a bbox label of an object. The object is
                    represented by a dict,
                which contains:
                 - bbox (list): bbox coordinates, top left and bottom right.
                 - bbox_mode (str): bbox label mode, for example: `XYXY_ABS`,
                    `XYWH_ABS` and so on...
        """
        sample_mode = (1, *self.min_ious, 0)
        h, w = img.shape[:2]

        boxes = torch.tensor(boxes)

        while True:
            mode = np.random.choice(sample_mode)
            if mode == 1:
                return NoOpTransform()

            min_iou = mode
            for _ in range(50):
                new_w = np.random.uniform(self.min_crop_size * w, w)
                new_h = np.random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = np.random.uniform(w - new_w)
                top = np.random.uniform(h - new_h)

                patch = torch.tensor([left, top, left + new_w, top + new_h], dtype=torch.int)

                overlaps = pairwise_iou(Boxes(patch.view(-1, 4)), Boxes(boxes.view(-1, 4)))

                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1])
                        * (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                if not mask.any():
                    continue
                return CropTransform(int(left), int(top), int(new_w), int(new_h))

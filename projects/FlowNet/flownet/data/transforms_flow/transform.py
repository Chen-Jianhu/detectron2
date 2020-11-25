# -*- coding: utf-8 -*-
import numpy as np
import torch

from fvcore.transforms.transform import Transform
from detectron2.data.transforms import (
    RotationTransform as _RotationTransform,
    CropTransform as _CropTransform
)

__all__ = [
    "TranslationTransform",
    "RotationTransform",
    "CropTransform",
]


class TranslationTransform(Transform):

    def __init__(self, trans_x: int, trans_y: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        h, w = img.shape[:2]
        x1, x2 = max(0, self.trans_x), min(w + self.trans_x, w)
        y1, y2 = max(0, self.trans_y), min(h + self.trans_y, h)
        img = img[y1: y2, x1: x2]

        # Save for apply_flow meth use
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        return img

    def apply_image2(self, img2: np.ndarray):
        h, w = img2.shape[:2]
        x1, x2 = max(0, -self.trans_x), min(w - self.trans_x, w)
        y1, y2 = max(0, -self.trans_y), min(h - self.trans_y, h)
        img2 = img2[y1: y2, x1: x2]
        return img2

    def apply_flow(self, flow: np.ndarray):
        flow = flow[self.y1: self.y2, self.x1: self.x2]
        flow[:, :, 0] += self.trans_x
        flow[:, :, 1] += self.trans_y
        return flow

    def apply_coords(self, coords: np.ndarray):
        raise ValueError("This is a flow transform!")


class RotationTransform(_RotationTransform):

    def apply_image2(self, img: np.ndarray):
        return self.apply_image(img)

    def apply_flow(self, flow: np.ndarray):
        flow = self.apply_image(flow)
        flow_ = np.copy(flow)

        angle_rad = np.deg2rad(self.angle)
        flow[:, :, 0] = np.cos(angle_rad) * flow_[:, :, 0] + np.sin(angle_rad) * flow_[:, :, 1]
        flow[:, :, 1] = -np.sin(angle_rad) * flow_[:, :, 0] + np.cos(angle_rad) * flow_[:, :, 1]

        return flow


class CropTransform(_CropTransform):

    def apply_image2(self, img: np.ndarray):
        return self.apply_image(img)

    def apply_flow(self, flow: np.ndarray):
        return self.apply_image(flow)


class FlipTransform(Transform):

    def __init__(self, direction: str):
        assert direction in ("horizontal", "vertical", "both"), direction

        do_horizontal = False
        do_vertical = False
        if direction == "horizontal" or direction == "both":
            do_horizontal = True
        if direction == "vertical" or direction == "both":
            do_vertical = True

        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        if self.do_horizontal:
            # NOTE: opencv would be faster:
            # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
            if img.ndim <= 3:  # HxW, HxWxC
                img = np.flip(img, axis=1)
            else:
                img = np.flip(img, axis=-2)

        if self.do_vertical:
            tensor = torch.from_numpy(np.ascontiguousarray(img))
            if len(tensor.shape) == 2:
                # For dimension of HxW.
                tensor = tensor.flip((-2))
            elif len(tensor.shape) > 2:
                # For dimension of HxWxC, NxHxWxC.
                tensor = tensor.flip((-3))
            img = tensor.numpy()

        return img

    def apply_image2(self, img: np.ndarray):
        return self.apply_image(img)

    def apply_flow(self, flow: np.ndarray):
        flow = self.apply_image(flow)

        if self.do_horizontal:
            flow[:, :, 0] *= -1

        if self.do_vertical:
            flow[:, :, 1] *= -1
        return flow

    def apply_coords(self, coords: np.ndarray):
        raise ValueError("This is a flow transform!")

# -*- coding: utf-8 -*-
import random
import numpy as np

from fvcore.transforms import NoOpTransform
from detectron2.data.transforms import (
    Augmentation,
    RandomRotation as _RandomRotation,
    RandomCrop as _RandomCrop,
)
from .transform import (
    TranslationTransform,
    RotationTransform,
    CropTransform,
    FlipTransform,
)

__all__ = [
    "RandomTranslation",
    "RandomRotation",
    "RandomCrop",
    "RandomFlip",
]


class RandomTranslation(Augmentation):

    def __init__(self, t_range: float):
        """
        Args:
            t_range (float):
            prob (float): probability of flip.
        """
        self._init(locals())
        assert 0 <= t_range <= 1., t_range

    def get_transform(self, image):
        h, w = image.shape[:2]
        trans_x = int(self.t_range * w)
        trans_y = int(self.t_range * h)
        trans_x = random.randint(-trans_x, trans_x + 1)
        trans_y = random.randint(-trans_y, trans_y + 1)
        if trans_x == trans_y == 0:
            return NoOpTransform()
        else:
            return TranslationTransform(trans_x, trans_y)


class RandomRotation(_RandomRotation):

    def get_transform(self, image):
        h, w = image.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class RandomCrop(_RandomCrop):

    def get_transform(self, image):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)


class RandomFlip(Augmentation):
    """
    Flip the image horizontallywith the given probability.
    """

    def __init__(self, direction, prob=0.5):
        """
        Args:
            prob (float): probability of flip.
            direction (str): optional as follows:
                * "horizontal": horizontal flip
                * "vertical": vertical flip
                * "both": both
        """
        assert direction in ("horizontal", "vertical", "both"), direction
        self._init(locals())

    def get_transform(self, image):
        if self.direction == "":
            return NoOpTransform()

        direction = self.direction
        if direction == "horizontal" or direction == "both":
            do_horizontal = (self._rand_range() < self.prob)
        if direction == "vertical" or direction == "both":
            do_vertical = (self._rand_range() < self.prob)

        if do_horizontal:
            direction = "horizontal"
        if do_vertical:
            direction = "vertical"
        if do_horizontal and do_vertical:
            direction = "both"

        return FlipTransform(direction)

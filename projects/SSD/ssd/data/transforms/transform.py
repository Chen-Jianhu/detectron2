import random
import cv2
import numpy as np

from fvcore.transforms.transform import Transform, CropTransform as _CropTransform


class ColorAugTransform(Transform):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        super().__init__()
        assert img_format in ["BGR", "RGB"]
        self.is_rgb = img_format == "RGB"
        del img_format
        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        img = self.brightness(img)
        if random.randrange(2):
            img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
        else:
            img = self.saturation(img)
            img = self.hue(img)
            img = self.contrast(img)
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.randrange(2):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta,
                                         self.brightness_delta)
            )
        return img

    def contrast(self, img):
        if random.randrange(2):
            return self.convert(img, alpha=random.uniform(self.contrast_low, self.contrast_high))
        return img

    def saturation(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=random.uniform(
                    self.saturation_low, self.saturation_high)
            )
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(
                    int) + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class RandomSwapChannelsTransform(Transform):
    """
    Randomly swap image channels.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        assert len(img.shape) > 2
        if random.randrange(2):
            img = img[..., np.random.permutation(3)]
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class ExpandTransform(Transform):
    """
    Expand the image and boxe  s according the specified expand ratio.
    """

    def __init__(self, ratio, top, left, img_value, seg_value):
        """
        Args:
            ratio_range (float): image expand ratio.
            img_value (tuple): mean value of dataset.
            seg_value:
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        """
        Randomly place the original image on a canvas of 'ratio' x original image
        size filled with mean values. The ratio is in the range of ratio_range.
        """
        h, w, c = img.shape
        expand_shape = (int(h * self.ratio), int(w * self.ratio), c)
        expand_img = np.full(expand_shape, self.img_value, dtype=img.dtype)
        expand_img[self.top: self.top + h, self.left: self.left + w] = img
        return expand_img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply expand transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            ndarray: expand coordinates.
        """
        coords[:, 0] += self.left
        coords[:, 1] += self.top
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        h, w = segmentation.shape
        expand_shape = (int(h * self.ratio), int(w * self.ratio))
        expand_seg = np.full(expand_shape, self.seg_value, dtype=segmentation.dtype)
        expand_seg[self.top: self.top + h, self.left: self.left + w] = segmentation
        return expand_seg

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class CropTransform(_CropTransform):

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box. By default will transform
        the corner points and use their minimum/maximum to create a new
        axis-aligned box. Note that this default may change the size of your
        box, e.g. after rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].

            This function does not clip boxes to force them inside the image.
            It is up to the application that uses the boxes to decide.
        """
        box = np.array(box).reshape(-1, 4)

        center = (box[:, :2] + box[:, 2:]) / 2
        mask = ((center[:, 0] > self.x0) * (center[:, 0] < self.x0 + self.w)
                * (center[:, 1] > self.y0) * (center[:, 1] < self.y0 + self.h))

        # Filter out boxes which center not in the croped region.
        if not mask.any():
            box = np.zeros_like(box)
            return box
        else:
            box = box[mask]

        return super().apply_box(box)

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

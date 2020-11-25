# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/projects/FlowNet/demo/predictor.py
@Time         : 2020-11-24 23:58:33
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-25 22:21:03
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

import torch
import numpy as np

from typing import List, Tuple
from detectron2.utils.flow_visualizer import flow2img
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_flow_net


class FlowPredictor:
    """
    Like DefaultPredictor.

    Examples:
    ::
        pred = FlowPredictor(cfg)
        image1 = cv2.imread("image1.jpg")
        image2 = cv2.imread("image2.jpg")
        inputs = [image1, image2]
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_flow_net(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    @torch.no_grad()
    def __call__(self, images: List[np.ndarray]):
        """
        Args:
            images (list[np.ndarray]): two images of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        assert isinstance(images, (Tuple, List))
        assert len(images) == 2, len(images)
        image1, image2 = images
        assert image1.shape == image2.shape, (
            f"Different shape between input images: {image1.shape} and {image2.shape}"
        )
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            image1 = image1[:, :, ::-1]
            image2 = image2[:, :, ::-1]

        height, width = image1.shape[:2]
        image1 = torch.as_tensor(image1.astype("float32").transpose(2, 0, 1))
        image2 = torch.as_tensor(image2.astype("float32").transpose(2, 0, 1))

        inputs = {"image1": image1, "image2": image2, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions


class VisualizationDemo(object):
    def __init__(self, cfg, parallel=False):
        """
        Args:
            cfg (CfgNode):
        """
        self.predictor = FlowPredictor(cfg)

    def _process_predictions(self, images: List[np.ndarray], predictions: dict):
        """
        Returns:
            output_image (np.ndarray): output_image is a RGB mode image.
        """
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image1 = images[0][:, :, ::-1]
        flow_image = flow2img(predictions["flow"].permute(1, 2, 0).detach().cpu().numpy())
        output_image = np.concatenate([image1, flow_image], axis=0)
        return output_image

    def run_on_images(self, images: List[np.ndarray]):
        """
        Args:
            images (list[np.ndarray]): two images of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            output_image (np.ndarray): the visualized image output.
        """
        predictions = self.predictor(images)
        output_image = self._process_predictions(images, predictions)
        return predictions, output_image

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        frame_gen = self._frame_from_video(video)
        frame1 = None
        frame2 = None
        for frame in frame_gen:
            if frame1 is None:
                frame1 = frame
                frame2 = frame
            else:
                frame1 = frame2
                frame2 = frame
            frames = [frame1, frame2]
            yield self._process_predictions(frames, self.predictor(frames))

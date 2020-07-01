# -*- encoding: utf-8 -*-
"""
@File          :   dff.py
@Time          :   2020/06/24 00:49:55
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:07:41
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.init import constant_

from detectron2.modeling import (
    META_ARCH_REGISTRY,
    build_backbone,
    detector_postprocess,
    build_proposal_generator,
    build_roi_heads
)
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm

from ..flownet import build_flownet
from ..nn_utils import crop_like

__all__ = ["DeepFeatureFlow"]


def get_flow_grid(flow):
    m, n = flow.shape[-2:]
    shifts_x = torch.arange(0, n, 1, dtype=torch.float32, device=flow.device)
    shifts_y = torch.arange(0, m, 1, dtype=torch.float32, device=flow.device)
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

    grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
    workspace = torch.tensor([(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)

    flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)

    return flow_grid


@META_ARCH_REGISTRY.register()
class DeepFeatureFlow(nn.Module):
    """
    Deep Feature Flow. See: for more details.
    Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Optical flow prediction (e.g. FlowNet)
    3. Region proposal generation
    4. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.flownet = build_flownet(cfg)
        self.scale_func = self.build_scale_function()
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        self.in_features = cfg.MODEL.RPN.IN_FEATURES

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        self.register_buffer(
            "flownet_pixel_mean", torch.Tensor(cfg.MODEL.FLOWNET.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer(
            "flownet_pixel_std", torch.Tensor(cfg.MODEL.FLOWNET.PIXEL_STD).view(-1, 1, 1))
        self.register_buffer("flow_div", torch.Tensor([cfg.MODEL.FLOWNET.FLOW_DIV]))
        # Delete redundent buffer
        del self.flownet.flownet_pixel_mean
        del self.flownet.flownet_pixel_std
        del self.flownet.flow_div

        # For testing
        self.key_frame_features = None
        self.key_frame_duration = cfg.MODEL.DFF.KEY_FRAME_DURATION

    @property
    def device(self):
        return self.pixel_mean.device

    def build_scale_function(self):
        scale_func = nn.Conv2d(194, 1024, 1, 1)
        constant_(scale_func.weight, 1)
        constant_(scale_func.bias, 0)
        return scale_func

    def forward_flownet(self, x):
        """
        Args:
            x (Tensor): concat image pairs with shape (N, 2*C, H, W).
        """
        # avg pool used for resize image pairs to half of the original resolution
        x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        # forward flownet
        out_conv2 = self.flownet.conv2(self.flownet.conv1(x))
        out_conv3 = self.flownet.conv3_1(self.flownet.conv3(out_conv2))
        out_conv4 = self.flownet.conv4_1(self.flownet.conv4(out_conv3))
        out_conv5 = self.flownet.conv5_1(self.flownet.conv5(out_conv4))
        out_conv6 = self.flownet.conv6_1(self.flownet.conv6(out_conv5))

        flow6 = self.flownet.predict_flow6(out_conv6)
        flow6_up = crop_like(self.flownet.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.flownet.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.flownet.predict_flow5(concat5)
        flow5_up = crop_like(self.flownet.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.flownet.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.flownet.predict_flow4(concat4)
        flow4_up = crop_like(self.flownet.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.flownet.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.flownet.predict_flow3(concat3)
        flow3_up = crop_like(self.flownet.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.flownet.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)

        # Special operations on DFF
        concat2 = F.avg_pool2d(concat2, kernel_size=2, stride=2, ceil_mode=True)

        flow = self.flownet.predict_flow2(concat2)
        scale_map = self.scale_func(concat2)
        # which is of the same spatial and channel dimensions as the feature maps
        # 1/16 of the original resolution

        return flow, scale_map

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image_cur"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=0)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        images_for_backbone = self.preprocess_image_for_backbone(batched_inputs)
        features = self.backbone(images_for_backbone.tensor)

        # feature propagation
        features = [features[k] for k in self.in_features]

        images_for_flownet = self.preprocess_image_for_flownet(batched_inputs)
        flow, scale_map = self.forward_flownet(images_for_flownet.tensor)

        flow_grid = get_flow_grid(flow)
        warped_features = [
            F.grid_sample(features_i, flow_grid, mode="bilinear", padding_mode="border")
            for features_i in features
        ]

        features = [
            warped_features_i * scale_map
            for warped_features_i in warped_features
        ]

        features = dict(zip(self.in_features, features))

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images_for_backbone, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images_for_backbone, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        assert comm.get_world_size() <= 1, (
            f"Cuda device count: {comm.get_word_size()}."
            "Only support use one cuda device when testing."
        )
        assert len(batched_inputs) == 1, "Only support one image for testing."

        frame_seg_id = batched_inputs[0]["frame_seg_id"]
        frame_seg_len = batched_inputs[0]["frame_seg_len"]
        # print("{} / {}".format(frame_seg_id, frame_seg_len))
        images_for_backbone = self.preprocess_image_for_backbone(batched_inputs)
        if frame_seg_id % self.key_frame_duration == 0:
            features = self.backbone(images_for_backbone.tensor)
            features = [features[k] for k in self.in_features]
            self.key_frame_features = features
        else:
            assert self.key_frame_features is not None
            features = self.key_frame_features

        if (frame_seg_id + 1) == frame_seg_len:
            self.key_frame_features = None

        images_for_flownet = self.preprocess_image_for_flownet(batched_inputs)
        flow, scale_map = self.forward_flownet(images_for_flownet.tensor)

        flow_grid = get_flow_grid(flow)
        warped_features = [
            F.grid_sample(features_i, flow_grid, mode="bilinear", padding_mode="border")
            for features_i in features
        ]
        features = [
            warped_features_i * scale_map
            for warped_features_i in warped_features
        ]
        features = dict(zip(self.in_features, features))

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images_for_backbone, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images_for_backbone, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return DeepFeatureFlow._postprocess(
                results, batched_inputs, images_for_backbone.image_sizes)
        else:
            return results

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def preprocess_image_for_backbone(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image_ref"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_image_for_flownet(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images1 = [x["image_ref"].to(self.device) for x in batched_inputs]
        images2 = [x["image_cur"].to(self.device) for x in batched_inputs]
        images = [torch.cat([img1, img2], dim=0) for img1, img2 in zip(images1, images2)]
        images = [
            (
                x / 255. - self.flownet_pixel_mean.repeat([2, 1, 1])
            ) / (
                self.flownet_pixel_std.repeat([2, 1, 1])
            )
            for x in images
        ]
        images = ImageList.from_tensors(images)
        return images

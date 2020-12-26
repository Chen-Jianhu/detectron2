# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/projects/DFF/dff/modeling/meta_arch/dff.py
@Time         : 2020-11-28 16:27:24
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-28 16:42:02
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.init import constant_
from detectron2.config import configurable
from detectron2.layers import cat
from detectron2.modeling import (
    Backbone,
    META_ARCH_REGISTRY,
    build_backbone,
    detector_postprocess,
    build_proposal_generator,
    build_roi_heads
)
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm
from detectron2.modeling.flow.flownets import crop_like
from detectron2.modeling import build_flow_net

__all__ = ["DeepFeatureFlow"]


@META_ARCH_REGISTRY.register()
class DeepFeatureFlow(nn.Module):
    """
    Deep Feature Flow. See: https://arxiv.org/abs/1611.07715 for more details.
    Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Optical flow prediction (e.g. FlowNet)
    3. Region proposal generation
    4. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        flow_net: nn.Module,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        flow_pixel_mean: Tuple[float],
        flow_pixel_std: Tuple[float],
        flow_div: float,
        input_format: Optional[str] = None,
        vis_period: int = 0,
        key_frame_duration: int = 10,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            flow_net: a optical flow prediction module
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()

        self.backbone = backbone
        self.flow_net = flow_net
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        # There is a special module for DFF network
        self.scale_func = self.build_scale_function()

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # There are special buffer for DFF network
        self.register_buffer("flow_pixel_mean", torch.Tensor(flow_pixel_mean).view(-1, 1, 1))
        self.register_buffer("flow_pixel_std", torch.Tensor(flow_pixel_std).view(-1, 1, 1))
        assert (
            self.flow_pixel_mean.shape == self.flow_pixel_std.shape
        ), f"{self.flow_pixel_mean} and {self.flow_pixel_std} have different shapes!"
        self.register_buffer("flow_div", torch.Tensor([flow_div]))

        # Delete redundent buffer
        del self.flow_net.flow_pixel_mean
        del self.flow_net.flow_pixel_std
        del self.flow_net.flow_div

        # For testing
        self.key_frame_features = None
        self.key_frame_duration = key_frame_duration

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "flow_net": build_flow_net(cfg),
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "flow_pixel_mean": cfg.MODEL.FLOW_NET.PIXEL_MEAN,
            "flow_pixel_std": cfg.MODEL.FLOW_NET.PIXEL_STD,
            "flow_div": cfg.MODEL.FLOW_NET.FLOW_DIV,
            "key_frame_duration": cfg.MODEL.DFF.KEY_FRAME_DURATION,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def build_scale_function(self):
        scale_func = nn.Conv2d(194, 1024, 1, 1)
        constant_(scale_func.weight, 1)
        constant_(scale_func.bias, 0)
        return scale_func

    def visualize_training(self, batched_inputs, proposals, pred_flows):
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
        from detectron2.utils.flow_visualizer import flow2img

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop, flow in zip(batched_inputs, proposals, pred_flows):
            img_ref = input["image_ref"]
            img_cur = input["image_cur"]
            img_ref = convert_image_to_rgb(img_ref.permute(1, 2, 0), self.input_format)
            img_cur = convert_image_to_rgb(img_cur.permute(1, 2, 0), self.input_format)

            v_gt = Visualizer(img_cur, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()

            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img_cur, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()

            flow = flow.detach().cpu()
            flow *= 8
            flow_resized = F.interpolate(flow.unsqueeze(
                dim=0), img_cur.shape[:2], mode='bilinear', align_corners=False)
            flow_img = flow2img(flow_resized.squeeze(dim=0).permute(1, 2, 0).numpy())

            vis_img_left = np.concatenate((img_ref, anno_img), axis=0)
            vis_img_right = np.concatenate((flow_img, prop_img), axis=0)
            vis_img = np.concatenate((vis_img_left, vis_img_right), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: Ref image and GT bounding boxes; Right: Predicted flow and proposals"
            storage.put_image(vis_name, vis_img)

            break  # only visualize one image in a batch

    def _get_flow_grid(self, flow):
        m, n = flow.shape[-2:]
        shifts_x = torch.arange(0, n, 1, dtype=torch.float32, device=flow.device)
        shifts_y = torch.arange(0, m, 1, dtype=torch.float32, device=flow.device)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

        grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
        workspace = torch.tensor([(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)

        flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)
        return flow_grid

    def forward_flow_net(self, x):
        """
        Args:
            x (Tensor): concat image pairs with shape (N, 2*C, H, W).
        """
        # https://github.com/msracver/Deep-Feature-Flow/blob/master/dff_rfcn/symbols/resnet_v1_101_flownet_rfcn.py#L482 # noqa
        # Original FlowNetS output is 1/4 of the input image pairs
        # avg pool used for resize image pairs to half of the original resolution
        x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        out_conv2 = self.flow_net.conv2(self.flow_net.conv1(x))
        out_conv3 = self.flow_net.conv3_1(self.flow_net.conv3(out_conv2))
        out_conv4 = self.flow_net.conv4_1(self.flow_net.conv4(out_conv3))
        out_conv5 = self.flow_net.conv5_1(self.flow_net.conv5(out_conv4))
        out_conv6 = self.flow_net.conv6_1(self.flow_net.conv6(out_conv5))

        flow6 = self.flow_net.pred6(out_conv6)

        flow6_up = crop_like(self.flow_net.upsample_flow6to5(flow6), out_conv5)
        out_deconv5 = crop_like(self.flow_net.deconv5(out_conv6), out_conv5)
        concat5 = cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.flow_net.pred5(concat5)

        flow5_up = crop_like(self.flow_net.upsample_flow5to4(flow5), out_conv4)
        out_deconv4 = crop_like(self.flow_net.deconv4(concat5), out_conv4)
        concat4 = cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.flow_net.pred4(concat4)

        flow4_up = crop_like(self.flow_net.upsample_flow4to3(flow4), out_conv3)
        out_deconv3 = crop_like(self.flow_net.deconv3(concat4), out_conv3)
        concat3 = cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.flow_net.pred3(concat3)

        flow3_up = crop_like(self.flow_net.upsample_flow3to2(flow3), out_conv2)
        out_deconv2 = crop_like(self.flow_net.deconv2(concat3), out_conv2)
        concat2 = cat((out_conv2, out_deconv2, flow3_up), 1)

        # Special operations on DFF
        concat2 = F.avg_pool2d(concat2, kernel_size=2, stride=2, ceil_mode=True)

        flow = self.flow_net.pred2(concat2)
        scale_map = self.scale_func(concat2)
        # which is of the same spatial and channel dimensions as the feature maps
        # 1/16 of the original resolution

        # https://github.com/msracver/Deep-Feature-Flow/issues/23
        # That's because the new FlowNetS (Trained by Chen-Jianhu) need
        # to multiply 10.0 to get the optical flow.
        # They divided the groundtruth optical flow by 10.0 during training.
        # Here we need optical flow with 1/8 input size of the flownet, so
        # instead of multiplying 10.0, we multiply it by 10.0 / 8.0 = 1.25
        # return flow, scale_map
        return flow * self.flow_div / 8.0, scale_map

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
        # "res4" resolution is 1/16 of the original image size

        images_for_flow_net = self.preprocess_image_for_flow_net(batched_inputs)
        flow, scale_map = self.forward_flow_net(images_for_flow_net.tensor)

        # feature propagation
        feature_names = list(features.keys())
        feature_values = list(features.values())

        flow_grid = self._get_flow_grid(flow)
        warped_features = [
            F.grid_sample(features_i, flow_grid, mode="bilinear", padding_mode="border")
            for features_i in feature_values
        ]

        features = [
            warped_features_i * scale_map
            for warped_features_i in warped_features
        ]

        features = dict(zip(feature_names, features))

        if self.proposal_generator is not None:
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
                self.visualize_training(batched_inputs, proposals, flow)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    @torch.no_grad()
    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
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
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        assert comm.get_world_size() <= 1, (
            f"Cuda device count: {comm.get_word_size()}."
            "Only support use one cuda device when testing."
        )
        assert len(batched_inputs) == 1, "Only support one image for testing."

        frame_seg_id = batched_inputs[0]["frame_seg_id"]
        frame_seg_len = batched_inputs[0]["frame_seg_len"]

        images_for_backbone = self.preprocess_image_for_backbone(batched_inputs)

        is_key_frame = (frame_seg_id % self.key_frame_duration == 0)
        if is_key_frame:
            assert self.key_frame_features is None, batched_inputs
            # s = time.time()
            self.key_frame_features = self.backbone(images_for_backbone.tensor)
            # print("🥁 Backbone: {}".format(time.time() - s))
            features = self.key_frame_features
        else:
            images_for_flow_net = self.preprocess_image_for_flow_net(batched_inputs)
            # s = time.time()
            flow, scale_map = self.forward_flow_net(images_for_flow_net.tensor)
            flow_grid = self._get_flow_grid(flow)
            # print("FlowNetS: {}".format(time.time() - s))
            warped_features = [
                F.grid_sample(features_i, flow_grid, mode="bilinear", padding_mode="border")
                for features_i in self.key_frame_features.values()
            ]
            features = [
                warped_features_i * scale_map
                for warped_features_i in warped_features
            ]
            features = dict(zip(self.key_frame_features.keys(), features))

        is_end_frame = ((frame_seg_id + 1) == frame_seg_len)
        next_frame_is_key = ((frame_seg_id + 1) % self.key_frame_duration == 0)
        if is_end_frame or next_frame_is_key:
            self.key_frame_features = None

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

    def preprocess_image_for_backbone(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image_ref"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_image_for_flow_net(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images1 = [x["image_ref"].to(self.device) for x in batched_inputs]
        images2 = [x["image_cur"].to(self.device) for x in batched_inputs]
        images = [cat([img1, img2], dim=0) for img1, img2 in zip(images1, images2)]
        images = [
            (x / 255. - self.flow_pixel_mean.repeat([2, 1, 1]))
            / (self.flow_pixel_std.repeat([2, 1, 1]))
            for x in images
        ]
        images = ImageList.from_tensors(images)
        return images

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

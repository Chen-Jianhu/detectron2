# -*- encoding: utf-8 -*-
"""
@File          :   flownets.py
@Time          :   2020/06/20 7:18:07
@Author        :   Chen-Jianhu (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:28:35
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple
from collections import OrderedDict
from fvcore.nn.weight_init import c2_msra_fill
from detectron2.layers import ConvTranspose2d, cat
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.flow_visualizer import flow2img
from detectron2.losses import endpoint_error
from detectron2.modeling import FLOW_NET_REGISTRY


__all__ = ["FlowNetS"]


def multiscale_endpoint_error(pred_flows, target_flow, weights=None):
    """
    Args:
        pred_flows (list[Tensor]): predicted multiscale flows.
        target_flow (Tensor): target flow.
        weights (list[float]): weight of endpoint error per scale.
    """
    def single_scale(pred_flow, target):
        """
        """
        h, w = pred_flow.shape[-2:]
        target_scaled = F.interpolate(target, (h, w), mode='area')
        return endpoint_error(pred_flow, target_scaled, reduction="mean")

    if not isinstance(pred_flows, (Tuple, List)):
        pred_flows = [pred_flows]

    assert(len(weights) == len(pred_flows))

    losses = 0.
    for pred_flow, weight in zip(pred_flows, weights):
        losses += weight * single_scale(pred_flow, target_flow)
    return losses


@FLOW_NET_REGISTRY.register()
class FlowNetS(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        negative_slope = cfg.MODEL.FLOW_NET.NEGATIVE_SLOPE
        self.multiscale_weights = cfg.MODEL.FLOW_NET.MULTISCALE_WEIGHTS
        # Vis parameters
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        self.register_buffer("flow_pixel_mean", torch.Tensor(
            cfg.MODEL.FLOW_NET.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("flow_pixel_std", torch.Tensor(
            cfg.MODEL.FLOW_NET.PIXEL_STD).view(-1, 1, 1))
        self.register_buffer("flow_div", torch.Tensor([cfg.MODEL.FLOW_NET.FLOW_DIV]))

        net_args = [
            # name | type | in_chs | out_chs| kernel_size | strid | pad
            ["conv1", "conv", 6, 64, 7, 2, 3],
            ["conv2", "conv", 64, 128, 5, 2, 2],
            ["conv3", "conv", 128, 256, 5, 2, 2],
            ["conv3_1", "conv", 256, 256, 3, 1, 1],
            ["conv4", "conv", 256, 512, 3, 2, 1],
            ["conv4_1", "conv", 512, 512, 3, 1, 1],
            ["conv5", "conv", 512, 512, 3, 2, 1],
            ["conv5_1", "conv", 512, 512, 3, 1, 1],
            ["conv6", "conv", 512, 1024, 3, 2, 1],
            ["conv6_1", "conv", 1024, 1024, 3, 1, 1],

            ["deconv5", "deconv", 1024, 512, 4, 2, 1],
            ["deconv4", "deconv", 1026, 256, 4, 2, 1],
            ["deconv3", "deconv", 770, 128, 4, 2, 1],
            ["deconv2", "deconv", 386, 64, 4, 2, 1],

            ["upsample_flow6to5", "deconv", 2, 2, 4, 2, 1],
            ["upsample_flow5to4", "deconv", 2, 2, 4, 2, 1],
            ["upsample_flow4to3", "deconv", 2, 2, 4, 2, 1],
            ["upsample_flow3to2", "deconv", 2, 2, 4, 2, 1],

            ["pred6", "conv", 1024, 2, 3, 1, 1],
            ["pred5", "conv", 1026, 2, 3, 1, 1],
            ["pred4", "conv", 770, 2, 3, 1, 1],
            ["pred3", "conv", 386, 2, 3, 1, 1],
            ["pred2", "conv", 194, 2, 3, 1, 1],
        ]
        self._build_layers(net_args, negative_slope)
        self._init_weights()

    def _build_layers(self, net_args, negative_slope):
        """
        Build layers for FlowNet.

        Args:
            net_args (list[list[str or int]]): contains some list,
                which every one specify one convolution layer parameters,
                the order as follows:
                [name | type | in_chs | out_chs| kernel_size | strid | pad].
                Note that the layer type must be `conv` or `deconv`.

            negative_slope (float): for LeakyRelu.
        """
        mapper = {
            "conv": nn.Conv2d,
            "deconv": ConvTranspose2d,
        }

        for layer_arg in net_args:
            name, conv_type = layer_arg[:2]
            assert conv_type in mapper, conv_type

            conv_args = layer_arg[2:]
            layer = OrderedDict(
                {"conv": mapper.get(conv_type)(*conv_args)}
            )

            if "pred" not in name and "upsample" not in name:
                layer["act"] = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

            self.add_module(name, nn.Sequential(layer))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                c2_msra_fill(m)
        # For BN layer, by default, the elements of :math:`\gamma` are set to 1
        # and the elements of :math:`\beta` are set to 0.

    def forward_layers(self, images):
        """
        Args：
            images (Tensor): shape is [N, 6, H, W].

        Returns:
            (tuple): a tuple which contains flow predictions, the order is
            flow2, flow3, flow4, flow5, flow6, they resolution decline by order.
        """
        out_conv2 = self.conv2(self.conv1(images))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.pred6(out_conv6)

        flow6_up = self.upsample_flow6to5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.pred5(concat5)

        flow5_up = self.upsample_flow5to4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.pred4(concat4)

        flow4_up = self.upsample_flow4to3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.pred3(concat3)

        flow3_up = self.upsample_flow3to2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.pred2(concat2)

        return flow2, flow3, flow4, flow5, flow6

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for two images.
                For now, each item in the list is a dict that contains:

                * image1, image2: Tensor, in (C, H, W) format.
                * flow_map: Tensor, in (C, H, W) format.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        multiscale_flows = self.forward_layers(images.tensor)

        if self.training:
            gt_flow = self.get_ground_truth(batched_inputs)
            losses = self.losses(multiscale_flows, gt_flow)
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(multiscale_flows, batched_inputs)
                    self.visualize_training(batched_inputs, results)
            return losses
        else:
            return self.inference(multiscale_flows, batched_inputs)

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize image pairs and flow map.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (list): a list that contains predicted flows. Both
                batched_inputs and proposals should have the same length.
        """
        assert len(batched_inputs) == len(results), (
            "Cannot visualize inputs and results of different sizes")
        storage = get_event_storage()

        for input_i, pred_flow_i in zip(batched_inputs, results):
            image1 = input_i["image1"]
            image1 = convert_image_to_rgb(image1.permute(1, 2, 0), self.input_format)
            image2 = input_i["image2"]
            image2 = convert_image_to_rgb(image2.permute(1, 2, 0), self.input_format)

            gt_flow_img = flow2img(input_i["flow_map"].permute(1, 2, 0).cpu().numpy())
            pred_flow_img = flow2img(pred_flow_i["flow"].permute(1, 2, 0).detach().cpu().numpy())

            assert(image1.shape == gt_flow_img.shape) and (gt_flow_img.shape == pred_flow_img.shape)

            h, w, c = image1.shape
            vis_img = np.zeros([2 * h, 2 * w, c], dtype=np.uint8)
            vis_img[:h, :w] = image1
            vis_img[h:, :w] = image2
            vis_img[:h, w:] = gt_flow_img
            vis_img[h:, w:] = pred_flow_img
            vis_img = vis_img.transpose(2, 0, 1)

            vis_name = "Left: image pairs; Top-Right: GT flow map; Bottom-Right: pred flow map"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    @torch.no_grad()
    def inference(self, multiscale_flows, batched_inputs):
        processed_results = []
        # flow2
        for flow_i, batched_input_i in zip(multiscale_flows[0], batched_inputs):
            h, w = batched_input_i["height"], batched_input_i["width"]
            flow_i = flow_i.unsqueeze(dim=0)
            flow_i = F.interpolate(flow_i, (h, w), mode='bilinear', align_corners=False)
            flow_i = flow_i.squeeze(dim=0)
            flow_i = self.flow_div * flow_i
            processed_results.append({"flow": flow_i})
        return processed_results

    def losses(self, pred_flows, target):
        h, w = target.shape[-2:]
        flow2 = F.interpolate(pred_flows[0], (h, w), mode='bilinear', align_corners=False)
        flow2_epe = self.flow_div * endpoint_error(flow2, target, reduction="mean")
        get_event_storage().put_scalar("EPE", flow2_epe)
        losses = multiscale_endpoint_error(pred_flows, target, weights=self.multiscale_weights)
        return {"loss": losses}

    @torch.no_grad()
    def get_ground_truth(self, batched_inputs):
        """
        Args:
            batched_inputs (list[dict])): see: method: `forward`.

        Returns:
            gt_flow_maps (Tensor):

        """
        tragets = self.preprocess_flow(batched_inputs)
        return torch.stack(tragets)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images1 = [x["image1"].to(self.device) for x in batched_inputs]
        images2 = [x["image2"].to(self.device) for x in batched_inputs]
        images = [cat([img1, img2], dim=0) for img1, img2 in zip(images1, images2)]
        images = [
            (
                x / 255. - self.flow_pixel_mean.repeat([2, 1, 1])
            ) / self.flow_pixel_std.repeat([2, 1, 1])
            for x in images
        ]
        images = ImageList.from_tensors(images)
        return images

    def preprocess_flow(self, batched_inputs):
        """
        Normalize the target flow.
        """
        flows = [x["flow_map"].to(self.device) for x in batched_inputs]
        flows = [x / self.flow_div for x in flows]
        return flows

    @property
    def device(self):
        return self.flow_pixel_mean.device

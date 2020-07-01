# -*- encoding: utf-8 -*-
"""
@File          :   flownets.py
@Time          :   2020/06/20 7:18:07
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:28:35
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.init import kaiming_normal_, constant_
from functools import partial
from detectron2.layers import Conv2d, get_norm
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb

from .build import FLOWNET_REGISTRY
from ..losses import multiscale_EPE, EPE
from ..nn_utils import Deconv2d, crop_like
from flownet.utils.visualizer import flow2img

__all__ = ["FlowNetS"]


@FLOWNET_REGISTRY.register()
class FlowNetS(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        norm = cfg.MODEL.FLOWNET.NORM
        negative_slope = cfg.MODEL.FLOWNET.NEGATIVE_SLOPE
        self.multiscale_weights = cfg.MODEL.FLOWNET.MULTISCALE_WEIGHTS
        # Vis parameters
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        self.register_buffer("flownet_pixel_mean", torch.Tensor(
            cfg.MODEL.FLOWNET.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("flownet_pixel_std", torch.Tensor(
            cfg.MODEL.FLOWNET.PIXEL_STD).view(-1, 1, 1))
        self.register_buffer("flow_div", torch.Tensor([cfg.MODEL.FLOWNET.FLOW_DIV]))

        self._init_layers(norm, negative_slope)
        self._init_weights()

    def _init_layers(self, norm, negative_slope):
        bias = (True if norm == "" else False)
        self.conv1 = Conv2d(6, 64, 7, 2, 3, bias=bias, norm=get_norm(norm, 64),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv2 = Conv2d(64, 128, 5, 2, 2, bias=bias, norm=get_norm(norm, 128),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv3 = Conv2d(128, 256, 5, 2, 2, bias=bias, norm=get_norm(norm, 256),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv3_1 = Conv2d(256, 256, 3, 1, 1, bias=bias, norm=get_norm(norm, 256),
                              activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv4 = Conv2d(256, 512, 3, 2, 1, bias=bias, norm=get_norm(norm, 512),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv4_1 = Conv2d(512, 512, 3, 1, 1, bias=bias, norm=get_norm(norm, 512),
                              activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv5 = Conv2d(512, 512, 3, 2, 1, bias=bias, norm=get_norm(norm, 512),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv5_1 = Conv2d(512, 512, 3, 1, 1, bias=bias, norm=get_norm(norm, 512),
                              activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv6 = Conv2d(512, 1024, 3, 2, 1, bias=bias, norm=get_norm(norm, 1024),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv6_1 = Conv2d(1024, 1024, 3, 1, 1, bias=bias, norm=get_norm(norm, 1024),
                              activation=nn.LeakyReLU(negative_slope, inplace=True))

        deconv2d = partial(Deconv2d, kernel_size=4, stride=2, padding=1, bias=False,
                           activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.deconv5 = deconv2d(1024, 512)
        self.deconv4 = deconv2d(1026, 256)
        self.deconv3 = deconv2d(770, 128)
        self.deconv2 = deconv2d(386, 64)

        predict_flow = partial(Conv2d, out_channels=2, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = deconv2d(2, 2)
        self.upsampled_flow5_to_4 = deconv2d(2, 2)
        self.upsampled_flow4_to_3 = deconv2d(2, 2)
        self.upsampled_flow3_to_2 = deconv2d(2, 2)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward_layers(self, images):
        """
        images (Tensor): shape is [N, 6, H, W]
        """
        out_conv2 = self.conv2(self.conv1(images))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        return flow2, flow3, flow4, flow5, flow6

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

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
        flow2_EPE = self.flow_div * EPE(flow2, target)
        get_event_storage().put_scalar("EPE", flow2_EPE)
        loss = multiscale_EPE(pred_flows, target, weights=self.multiscale_weights)
        return {"loss": loss}

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
        images = [torch.cat([img1, img2], dim=0) for img1, img2 in zip(images1, images2)]
        images = [
            (
                x / 255. - self.flownet_pixel_mean.repeat([2, 1, 1])
            ) / self.flownet_pixel_std.repeat([2, 1, 1])
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
        return self.flownet_pixel_mean.device

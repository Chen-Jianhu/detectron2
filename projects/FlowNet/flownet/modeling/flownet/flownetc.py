# -*- encoding: utf-8 -*-
"""
@File          :   flownetc.py
@Time          :   2020/06/20 7:19:58
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:28:27
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import torch
import torch.nn as nn

from functools import partial
from torch.nn.init import kaiming_normal_, constant_
from detectron2.layers import Conv2d, get_norm

from .build import FLOWNET_REGISTRY
from ..nn_utils import Deconv2d, correlate, crop_like

__all__ = ["FlowNetC"]


@FLOWNET_REGISTRY.register()
class FlowNetC(nn.Module):

    def __init__(self, norm="BN", negative_slope=0.1):
        super().__init__()

        bias = (True if norm == "" else False)
        self.conv1 = Conv2d(3, 64, 7, 2, bias=bias, norm=get_norm(norm, 64),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv2 = Conv2d(64, 128, 5, 2, bias=bias, norm=get_norm(norm, 128),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv3 = Conv2d(128, 256, 5, 2, bias=bias, norm=get_norm(norm, 256),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv_redir = Conv2d(256, 32, 1, 1, bias=bias, norm=get_norm(norm, 32),
                                 activation=nn.LeakyReLU(negative_slope, inplace=True))

        self.conv3_1 = Conv2d(473, 256, 3, 1, bias=bias, norm=get_norm(norm, 256),
                              activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv4 = Conv2d(256, 512, 3, 2, bias=bias, norm=get_norm(norm, 512),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv4_1 = Conv2d(512, 512, 3, 1, bias=bias, norm=get_norm(norm, 512),
                              activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv5 = Conv2d(512, 512, 3, 2, bias=bias, norm=get_norm(norm, 512),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv5_1 = Conv2d(512, 512, 3, 1, bias=bias, norm=get_norm(norm, 512),
                              activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv6 = Conv2d(512, 1024, 3, 2, bias=bias, norm=get_norm(norm, 1024),
                            activation=nn.LeakyReLU(negative_slope, inplace=True))
        self.conv6_1 = Conv2d(1024, 1024, 3, 1, bias=bias, norm=get_norm(norm, 1024),
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

        self.upsampled_flow6_to_5 = Deconv2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = Deconv2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = Deconv2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = Deconv2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        x1 = x[:, :3]
        x2 = x[:, 3:]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        out_conv_redir = self.conv_redir(out_conv3a)
        out_correlation = correlate(out_conv3a, out_conv3b)

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(in_conv3_1)
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
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2a)

        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

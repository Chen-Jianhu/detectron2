# -*- encoding: utf-8 -*-
"""
@File          :   nn_utils.py
@Time          :   2020/06/27 20:54:14
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:07:50
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import torch.nn.functional as F

from detectron2.layers import ConvTranspose2d
from spatial_correlation_sampler import spatial_correlation_sample


class Deconv2d(ConvTranspose2d):

    def __init__(self, *args, **kwargs):
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w) / input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)

# -*- encoding: utf-8 -*-
"""
@File          :   endpoint_error.py
@Time          :   2020/06/27 20:53:53
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:07:27
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


def EPE(input_flow, target_flow, mean=True):
    """
    """
    EPE_map = torch.norm(target_flow - input_flow, p=2, dim=-3)

    if mean:
        return EPE_map.mean()
    else:
        return EPE_map


def multiscale_EPE(pred_flows, target_flow, weights=None):
    """

    Args:
        pred_flows (list[Tensor]): predicted multiscale flows.
        target_flow (Tensor): target flow.
        weights (list[float]): weight of EEP per scale.
    """
    def single_scale(pred_flow, target):
        """
        """
        h, w = pred_flow.shape[-2:]
        target_scaled = F.interpolate(target, (h, w), mode='area')
        return EPE(pred_flow, target_scaled, mean=False)

    if not isinstance(pred_flows, (Tuple, List)):
        pred_flows = [pred_flows]

    assert(len(weights) == len(pred_flows))

    losses = []
    for pred_flow, weight in zip(pred_flows, weights):
        loss = weight * single_scale(pred_flow, target_flow).sum() / pred_flow.size(0)
        losses.append(loss)
    return sum(losses)

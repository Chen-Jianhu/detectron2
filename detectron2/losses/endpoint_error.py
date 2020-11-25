# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/detectron2/losses/endpoint_error.py
@Time         : 2020-11-24 23:58:33
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-25 22:25:54
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

import torch


def endpoint_error(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "none"
) -> torch.Tensor:
    """
    End-to-end point error is calculated by comparing an estimate optical flow
    vector (v_est) with a groundtruth optical flow vector (v_gt).

    End-to-end point error is defined as the Euclidean distance between these two:

        || v_gt - v_est ||

    Args:
        input (Tensor): estimate optical flow vector.
        target (Tensor): groundtruth optical flow vector.
        reduction (str): 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.

    """
    error = torch.norm(target - input, p=2, dim=-3)

    if reduction == "mean":
        error = error.mean()
    elif reduction == "sum":
        error = error.sum()

    return error

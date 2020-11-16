# -*- encoding: utf-8 -*-
"""
@File    :   convert_darknet_weights.py
@Time    :   2020/10/27 14:08:30
@Author  :   Jianhu Chen
@E-mail  :   jhchen.mail@gmail.com
@License :   Copyright(C), USTC
@Desc    :

First, you should download darknet53_448.weights:
https://pjreddie.com/media/files/darknet53_448.weights (159M)

Then, run command:
  python convert_darknet.py [/path/to/darknet53_448.weights] [/path/to/output/dir]

Ref: https://github.com/yqyao/YOLOv3_Pytorch/blob/master/convert_darknet.py
"""

import os
import sys
import numpy as np
import torch
import copy
import pickle as pkl

from typing import List, OrderedDict
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.layers import Conv2d, BatchNorm2d, FrozenBatchNorm2d

sys.path.insert(0, "..")  # noqa
from yolov3 import add_yolov3_config


if __name__ == "__main__":
    assert len(sys.argv) == 3, __doc__
    file_path = sys.argv[1]
    output_dir = sys.argv[2]

    config_path = "../configs/Base-YOLOv3-Darknet53.yaml"
    cfg = get_cfg()
    add_yolov3_config(cfg)
    cfg.merge_from_file(config_path)
    darknet53 = build_backbone(cfg)

    # Read file
    with open(file_path, "rb") as fp:
        # The first 5 values are header information:
        # 1. Major version number (int32)
        # 2. Minor Version Number (int32)
        # 3. Subversion number    (int32)
        # 4. Images seen          (int64)
        version = np.fromfile(fp, dtype=np.int32, count=3)
        seen = np.fromfile(fp, dtype=np.int64, count=1)
        weights = np.fromfile(fp, dtype=np.float32)

    model = OrderedDict()

    def parse_module(
        prefix: List[str],
        module: torch.nn.Module,
        weights: np.ndarray,
        ptr: int
    ) -> int:
        if isinstance(module, torch.nn.Linear):
            num_biases = module.bias.numel()
            biases = torch.from_numpy(weights[ptr:ptr + num_biases])
            biases = biases.view_as(module.bias.data)
            ptr += num_biases

            num_weights = module.weight.numel()
            linear_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
            linear_weights = linear_weights.view_as(module.weight.data)
            ptr += num_weights

            model[".".join(prefix + ["bias"])] = biases
            model[".".join(prefix + ["weight"])] = linear_weights
        elif isinstance(module, Conv2d):
            # Parse BN params
            bn = module.norm
            assert isinstance(bn, (BatchNorm2d, FrozenBatchNorm2d)), bn

            num_bn_biases = bn.bias.numel()

            bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
            ptr += num_bn_biases

            bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
            ptr += num_bn_biases

            bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
            ptr += num_bn_biases

            bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
            ptr += num_bn_biases

            # Cast the loaded weights into dims of model weights.
            bn_biases = bn_biases.view_as(bn.bias.data)
            bn_weights = bn_weights.view_as(bn.weight.data)
            bn_running_mean = bn_running_mean.view_as(bn.running_mean)
            bn_running_var = bn_running_var.view_as(bn.running_var)

            # Parse conv params
            num_weights = module.weight.numel()
            conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
            ptr = ptr + num_weights

            conv_weights = conv_weights.view_as(module.weight.data)

            model[".".join(prefix + ["weight"])] = conv_weights
            model[".".join(prefix + ["norm", "bias"])] = bn_biases
            model[".".join(prefix + ["norm", "weight"])] = bn_biases
            model[".".join(prefix + ["norm", "running_mean"])] = bn_biases
            model[".".join(prefix + ["norm", "running_var"])] = bn_biases
            model[".".join(prefix + ["norm", "num_batches_tracked"])
                  ] = torch.tensor(900000., dtype=torch.float32)
        else:
            for name, children in module.named_children():
                prefix_copy = copy.deepcopy(prefix)
                prefix_copy.append(name)
                ptr = parse_module(prefix_copy, children, weights, ptr)

        return ptr

    ptr = parse_module([], darknet53, weights, 0)
    print("Rest               = {}".format(len(weights) - ptr))
    print("1024 * 1000 + 1000 = {}".format(1024 * 1000 + 1000))

    res = {
        "model": model,
        "matching_heuristics": True,
        "__author__": "Joseph Redmon",
        "__convert__": "Chen-Jianhu (jhchen.mail@gmail.com)",
        "__license__": "Copyright(C), USTC",
    }

    if output_dir.endswith("pkl"):
        output_path = output_dir
    else:
        output_path = os.path.join(output_dir, "darknet53_448.pkl")

    with open(output_path, "wb") as fp:
        pkl.dump(res, fp)

    print("Done!")

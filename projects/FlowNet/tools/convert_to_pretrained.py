# -*- encoding: utf-8 -*-
"""
@File          :   convert_to_pretrained.py
@Time          :   2020/06/28 3:09:38
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:29:42
@License       :   Copyright(C), USTC
@Desc          :   Used for convert trained model to pretrained,
                   they different is pretrained only keep model's
                   trained parameters and model infos.

Usage:
    python convert_to_pretrained.py  \\
        /path/to/cfg/file \\
        /path/to/trained/model

Examples:
    python convert_to_pretrained.py \\
        ../configs/flownets.yaml \\
        ../labs/flownets_bs32_iter500k/output/model_final.pth
"""

# here put the import lib
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.abspath(__file__))[:-len("tools")])  # noqa
import torch
import pickle as pkl

from detectron2.config import get_cfg
from flownet.config import add_flownet_config


def convert_to_pretrained_model(cfg, model_path):
    res = {
        "format": cfg.INPUT.FORMAT,
        "matching_heuristics": True,
        "__author__": "Jianhu Chen (jhchen.mail@gmail.com)",
        "__license__": "Copyright(C), USTC",
    }
    print(res)

    obj = torch.load(model_path, map_location="cpu")
    obj = obj["model"]

    newmodel = {}
    for k, v in obj.items():
        print(k)
        newmodel[k] = v.numpy()

    res["model"] = newmodel

    assert model_path.endswith(".pth")
    save_dir = model_path.replace(".pth", ".pkl")

    with open(save_dir, "wb") as f:
        pkl.dump(res, f)

    print(f"Save to {save_dir}.")
    print("Done.")


if __name__ == "__main__":
    assert len(sys.argv[1:]) == 2, __doc__

    (cfg_file_path, model_path) = sys.argv[1:]

    cfg = get_cfg()
    add_flownet_config(cfg)
    cfg.merge_from_file(cfg_file_path)
    cfg.freeze()

    convert_to_pretrained_model(cfg, model_path)

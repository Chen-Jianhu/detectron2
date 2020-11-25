# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/tools/convert_to_pretrained.py
@Time         : 2020-11-25 23:24:28
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-25 23:25:07
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : Used for convert trained model to pretrained,
                they different is pretrained only keep model's
                trained parameters and model infos.
Usage:
    python convert_to_pretrained.py  /path/to/trained/model $(input_format) [$(matching_heuristics)]

Examples:
    python convert_to_pretrained.py ../model_final.pth BGR
"""

import os
import sys
import torch
import pickle as pkl


def convert_to_pretrained_model(
    model_path,
    *,
    input_format=None,
    matching_heuristics=None,
):
    assert model_path.endswith(".pth"), model_path
    assert input_format in ("RGB", "BGR", None), input_format
    assert matching_heuristics in (True, False, None), matching_heuristics

    res = {
        "__author__": "Chen-Jianhu (jhchen.mail@gmail.com)",
        "__license__": "Copyright(C), USTC",
    }

    if input_format is not None:
        res["format"] = input_format
    if matching_heuristics is not None:
        res["matching_heuristics"] = matching_heuristics

    print(res)

    obj = torch.load(model_path, map_location="cpu")
    obj = obj["model"]

    model = {}
    for k, v in obj.items():
        print(k)
        model[k] = v.numpy()

    res["model"] = model

    save_dir = model_path.replace(".pth", ".pkl")
    with open(save_dir, "wb") as f:
        pkl.dump(res, f)

    print(f"Save to :{os.path.abspath(save_dir)}.")
    print("Done.")


if __name__ == "__main__":
    assert 3 <= len(sys.argv) <= 4, __doc__

    model_path = sys.argv[1]
    input_format = sys.argv[2].upper()
    matching_heuristics = sys.argv[3].lower() if len(sys.argv) == 4 else None

    if len(sys.argv) == 4:
        if "true" in matching_heuristics:
            matching_heuristics = True
        elif "false" in matching_heuristics:
            matching_heuristics = False

    convert_to_pretrained_model(model_path, input_format=input_format,
                                matching_heuristics=matching_heuristics)

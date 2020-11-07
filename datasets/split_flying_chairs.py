# -*- encoding: utf-8 -*-
"""
@File          :   split_flying_chairs.py
@Time          :   2020/07/01 8:31:48
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Chen-Jianhu (jhchen.mail@gmail.com)
@Last Modified :   2020/11/07 21:22:58
@License       :   Copyright(C), USTC
@Desc          :   This script is used for split Flying Chairs dataset.

Usage:
    python split_flying_chairs.py
"""

import os.path as osp
import glob
import json
import numpy as np
import time


def _generate_image_flow_map_pairs(root):
    start = time.time()
    print("Start generating image and flow map pairs ...")
    pairs = []
    num_unvalid = 0
    data_root = osp.join(root, "data")
    for flow_map in sorted(glob.glob(osp.join(data_root, "*_flow.flo"))):
        flow_map_file_name = osp.basename(flow_map)
        name_prefix = flow_map_file_name[: -len("_flow.flo")]
        img1_path = osp.join(data_root, name_prefix + '_img1.ppm')
        img2_path = osp.join(data_root, name_prefix + '_img2.ppm')

        if not (osp.isfile(img1_path) and osp.isfile(img2_path)):
            num_unvalid += 1
            continue

        pairs.append({
            "image_file1": img1_path,
            "image_file2": img2_path,
            "flow_map_file": flow_map
        })
    print("Finished. time: {:.2f}s.".format(time.time() - start))
    print("Total number of samples: {} (unvalid samples: {}).".format(len(pairs), num_unvalid))
    return pairs


def _random_split(pairs, split_ratio: float = 0.972):
    """
    Args:
        pairs (list): sample list, each item like this:
            {
                "image1": img1_path,
                "image2": img2_path,
                "flow_map": flow_map
            }
        split_ratio (tuple): the training set ratio,
            so test set ration is compute by `1 - split_ratio`.
            # This default split ratios if refrence the FlowNet paper.
    """
    start = time.time()
    print(
        "Start to randomly divide the samples into training and test sets, "
        "training set ratio: {} ...".format(split_ratio)
    )
    train_set_idxs = np.random.uniform(0, 1, len(pairs)) < split_ratio
    train_set = []
    test_set = []
    for train_set_idx, sample in zip(train_set_idxs, pairs):
        if train_set_idx:
            train_set.append(sample)
        else:
            test_set.append(sample)
    print("Finished. time: {:.2f}s.".format(time.time() - start))
    print("Train: {}, Test: {}.".format(len(train_set), len(test_set)))
    return train_set, test_set


if __name__ == "__main__":
    dataset_dir = osp.abspath("FlyingChairs_release")
    print("Dataset root: {}".format(dataset_dir))
    image_flow_map_pairs = _generate_image_flow_map_pairs(dataset_dir)
    datasets = _random_split(image_flow_map_pairs)

    # Dump to json file
    for s, dataset in zip(["train", "test"], datasets):
        file_name = "flying_chairs_{}.json".format(s)
        file_path = osp.join(dataset_dir, file_name)
        start = time.time()
        print("Start writing {} to {} ...".format(file_name, file_path))
        json_str = json.dumps(dataset, ensure_ascii=False, indent=4)
        with open(file_path, "w") as f:
            f.write(json_str)
        print("Finished. time: {:.2f}s.".format(time.time() - start))

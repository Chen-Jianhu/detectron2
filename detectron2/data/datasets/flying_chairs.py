# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/detectron2/data/datasets/flying_chairs.py
@Time         : 2020-11-24 23:58:33
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-25 22:24:41
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

import os.path as osp
import logging
import json

from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog


logger = logging.getLogger(__name__)


def register_flying_chairs(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_flying_chairs_json(name, json_file, image_root))
    MetadataCatalog.get(name).set(**metadata)


def get_flying_chairs_meta():
    meta = {"evaluator_type": "flying_chairs"}
    return meta


def load_flying_chairs_json(dataset_name, json_file, image_root):
    with PathManager.open(json_file) as f:
        json_str = f.read()
    dataset_dicts = json.loads(json_str)

    for idx in range(len(dataset_dicts)):
        dataset_dicts[idx]["image_file1"] = osp.join(
            image_root, dataset_dicts[idx]["image_file1"])
        dataset_dicts[idx]["image_file2"] = osp.join(
            image_root, dataset_dicts[idx]["image_file2"])
        dataset_dicts[idx]["flow_map_file"] = osp.join(
            image_root, dataset_dicts[idx]["flow_map_file"])

    logger.info(
        "Loaded {} image pairs and flow map from {}.".format(
            len(dataset_dicts), json_file
        )
    )
    return dataset_dicts


if __name__ == "__main__":
    import cv2
    from PIL import Image
    import numpy as np
    from detectron2.utils.logger import setup_logger
    from detectron2.data.datasets.builtin import _PREDEFINED_SPLITS_FLYING_CHAIRS

    logger = setup_logger(name=__name__)

    root = "/data/datasets/"
    name = "flying_chairs_train"
    json_file = _PREDEFINED_SPLITS_FLYING_CHAIRS[name]
    json_file = root + json_file

    dicts = load_flying_chairs_json(name, json_file)

    num_samples = 5
    for sample in dicts[:num_samples]:
        image1 = np.array(Image.open(sample["image_file1"]))
        image2 = np.array(Image.open(sample["image_file2"]))
        cv2.imshow("image1", image1[:, :, ::-1])
        cv2.imshow("image2", image1[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

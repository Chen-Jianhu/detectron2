# -*- encoding: utf-8 -*-
"""
@File          :   imagenet_vid.py
@Time          :   2020/06/22 12:14:36
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:05:44
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import os
import pickle
import logging
import numpy as np
import multiprocessing as mp
import xml.etree.ElementTree as ET

from typing import List
from functools import partial

from fvcore.common.file_io import PathManager
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils import comm

logger = logging.getLogger(__name__)


CLASS_NAME_MAPPER = dict(zip(
    [
        'n02691156', 'n02419796', 'n02131653', 'n02834778',
        'n01503061', 'n02924116', 'n02958343', 'n02402425',
        'n02084071', 'n02121808', 'n02503517', 'n02118333',
        'n02510455', 'n02342885', 'n02374451', 'n02129165',
        'n01674464', 'n02484322', 'n03790512', 'n02324045',
        'n02509815', 'n02411705', 'n01726692', 'n02355227',
        'n02129604', 'n04468005', 'n01662784', 'n04530566',
        'n02062744', 'n02391049'
    ],
    [
        'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle',
        'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 'horse',
        'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
        'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra'
    ]
))


def register_imagenet_vid_instances(name, metadata, img_dir, anno_dir, img_index):
    DatasetCatalog.register(
        name, lambda: load_imagenet_vid_instances(name, img_dir, anno_dir, img_index))
    MetadataCatalog.get(name).set(
        img_dir=img_dir, anno_dir=anno_dir, img_index=img_index, **metadata)


def get_imagenet_vid_instances_meta():
    # Mapping from the category id to an id in [0, 29]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(CLASS_NAME_MAPPER.keys())}
    meta = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": list(CLASS_NAME_MAPPER.values()),
        "evaluator_type": "imagenet_vid",
    }
    return meta


def _filter_and_cache_keep_infos(dataset_name, image_set_index, anno_dir, cache_dir):
    anno_file = os.path.join(anno_dir, "{}.xml")
    cache_file = os.path.join(cache_dir, dataset_name + "_keep.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as fid:
            keep = pickle.load(fid)
        logger.info(
            "Load {}'s keep annotations information from {}.".format(dataset_name, cache_file))
    else:
        logger.info("Start filtering out invalid annotations ...")
        num_images = len(image_set_index)
        keep = np.zeros((num_images), dtype=np.bool)
        pgbar = tqdm(range(num_images))
        for idx in pgbar:
            filename = image_set_index[idx]
            anno_file_i = anno_file.format(filename)
            tree = ET.parse(anno_file_i).getroot()
            objs = tree.findall("object")
            if len(objs) == 0:
                keep[idx] = False
            else:
                keep[idx] = True

        with open(cache_file, "wb") as fid:
            pickle.dump(keep, fid)
        logger.info(
            "Saving {}'s keep annotations information into {}.".format(dataset_name, cache_file))

    num_images_without_valid_annotations = np.sum(~keep)
    if num_images_without_valid_annotations > 0:
        logger.warning(
            "Filtered out {} images without valid annotations. "
            "There might be issues in your dataset generation process.".format(
                num_images_without_valid_annotations
            )
        )
    return keep


def _load_one_dict(
    idx,
    is_vid_set: bool,
    class_ids: list,
    img_dir: str,
    anno_dir: str,
    image_set_index: List[str],
    frame_id: List[str],
    pattern: List[str],
    frame_seg_id: List[str],
    frame_seg_len: List[str]
):
    img_file_i = os.path.join(img_dir, "{}.JPEG".format(image_set_index[idx]))
    anno_file_i = os.path.join(anno_dir, "{}.xml".format(image_set_index[idx]))

    with PathManager.open(anno_file_i) as f:
        tree = ET.parse(f)

    r = {
        "file_name": img_file_i,
        "image_id": image_set_index[idx],
        "height": int(tree.findall("./size/height")[0].text),
        "width": int(tree.findall("./size/width")[0].text),
        "frame_id": frame_id[idx],
    }

    # VID set
    if is_vid_set:
        r.update({
            "pattern": pattern[idx],
            "frame_seg_id": frame_seg_id[idx],
            "frame_seg_len": frame_seg_len[idx],
        })

    instances = []
    for obj in tree.findall("object"):
        cls = obj.find("name").text.lower().strip()
        if cls not in class_ids:
            # Only load 30 class annotations (VID dataset)
            continue

        bbox = obj.find("bndbox")
        bbox = [
            max(float(bbox.find("xmin").text), 0),
            max(float(bbox.find("ymin").text), 0),
            min(float(bbox.find("xmax").text), r["width"] - 1),
            min(float(bbox.find("ymax").text), r["height"] - 1),
        ]
        instances.append(
            {
                "category_id": class_ids.index(cls),
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS
            }
        )
    r["annotations"] = instances
    return r


def _load_and_cache_annotations_mp(
    dataset_name: str,
    img_dir: str,
    anno_dir: str,
    image_set_index: List[str],
    frame_id: List[str],
    pattern: List[str],
    frame_seg_id: List[str],
    frame_seg_len: List[str],
    cache_dir: str = None,
):
    """
    Args:

    Returns:
        dicts (list[dict]):
    """
    assert (
        len(image_set_index) == len(frame_id)
    ), "Error: image_set_index and others have differient length."

    is_vid_set = False
    if pattern is not None:
        assert (
            len(image_set_index) == len(pattern) == len(frame_seg_id) == len(frame_seg_len)
        ), "Error: image_set_index and others have differient length."
        is_vid_set = True

    cache_file = os.path.join(cache_dir, dataset_name + "_anno.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as fid:
            dicts = pickle.load(fid)
        logger.info("Load {}'s annotations from {}".format(dataset_name, cache_file))
    else:
        class_ids = list(CLASS_NAME_MAPPER.keys())
        processes = max(mp.cpu_count() // comm.get_world_size() // 2, 4)
        pool = mp.Pool(processes=processes)
        func = partial(
            _load_one_dict,
            is_vid_set=is_vid_set,
            class_ids=class_ids,
            img_dir=img_dir,
            anno_dir=anno_dir,
            image_set_index=image_set_index,
            frame_id=frame_id,
            pattern=pattern,
            frame_seg_id=frame_seg_id,
            frame_seg_len=frame_seg_len
        )
        dicts = list(tqdm(
            pool.imap(func, list(range(len(image_set_index)))),
            total=len(image_set_index),
            desc="Used processes: {}".format(processes)))

        with open(cache_file, "wb") as fid:
            pickle.dump(dicts, fid)
        logger.info("Saving {}'s annotation dicts to {}.".format(dataset_name, cache_file))

    # It is important when laoding validation set or test set.
    if is_vid_set:
        dicts = sorted(dicts, key=lambda x: x["frame_id"])

    logger.info("Done loading {} samples.".format(len(dicts)))
    return dicts


def _load_and_cache_annotations(
    dataset_name: str,
    img_dir: str,
    anno_dir: str,
    image_set_index: List[str],
    frame_id: List[str],
    pattern: List[str],
    frame_seg_id: List[str],
    frame_seg_len: List[str],
    cache_dir: str = None,
):
    """
    Args:

    Returns:
        dicts (list[dict]):
    """
    assert (
        len(image_set_index) == len(frame_id)
    ), "Error: image_set_index and others have differient length."

    is_vid_set = False
    if pattern is not None:
        assert (
            len(image_set_index) == len(pattern) == len(frame_seg_id) == len(frame_seg_len)
        ), "Error: image_set_index and others have differient length."
        is_vid_set = True

    cache_file = os.path.join(cache_dir, dataset_name + "_anno.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as fid:
            dicts = pickle.load(fid)
        logger.info("Load {}'s annotations from {}".format(dataset_name, cache_file))
    else:
        dicts = []
        jpeg_file = os.path.join(img_dir, "{}.JPEG")
        anno_file = os.path.join(anno_dir, "{}.xml")
        class_ids = list(CLASS_NAME_MAPPER.keys())

        num_images = len(image_set_index)
        pgbar = tqdm(range(num_images))
        for idx in pgbar:
            img_file_i = jpeg_file.format(image_set_index[idx])
            anno_file_i = anno_file.format(image_set_index[idx])

            with PathManager.open(anno_file_i) as f:
                tree = ET.parse(f)

            r = {
                "file_name": img_file_i,
                "image_id": image_set_index[idx],
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
                "frame_id": frame_id[idx],
            }

            # VID set
            if is_vid_set:
                r.update({
                    "pattern": pattern[idx],
                    "frame_seg_id": frame_seg_id[idx],
                    "frame_seg_len": frame_seg_len[idx],
                })

            instances = []
            for obj in tree.findall("object"):
                cls = obj.find("name").text.lower().strip()
                if cls not in class_ids:
                    # Only load 30 class annotations (VID dataset)
                    continue

                bbox = obj.find("bndbox")
                bbox = [
                    max(float(bbox.find("xmin").text), 0),
                    max(float(bbox.find("ymin").text), 0),
                    min(float(bbox.find("xmax").text), r["width"] - 1),
                    min(float(bbox.find("ymax").text), r["height"] - 1),
                ]
                instances.append(
                    {
                        "category_id": class_ids.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS
                    }
                )
            r["annotations"] = instances
            dicts.append(r)

        with open(cache_file, "wb") as fid:
            pickle.dump(dicts, fid)
        logger.info("Saving {}'s annotation dicts to {}".format(dataset_name, cache_file))

    # It is important when laoding validation set or test set.
    if is_vid_set:
        dicts = sorted(dicts, lambda x: x["frame_id"])

    logger.info("Done loading {} samples.".format(len(dicts)))
    return dicts


def load_imagenet_vid_instances(
    dataset_name: str,
    img_dir: str,
    anno_dir: str,
    img_index: str
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dataset_name (str): Contain "Annotations", "ImageSets", "JPEGImages"
        img_dir (str): one of "train", "test", "val", "trainval"
        anno_dir (str):
        img_index (int):

    Returns:
        dicts (list[list[dict]]):
    """
    with PathManager.open(img_index) as f:
        lines = [x.strip().split(" ") for x in f.readlines()]

    pattern = None
    frame_seg_id = None
    frame_seg_len = None
    # DET set
    if len(lines[0]) == 2:
        image_set_index = [x[0] for x in lines]
        frame_id = [int(x[1]) for x in lines]
    # VID set
    else:
        image_set_index = ["%s/%06d" % (x[0], int(x[2])) for x in lines]
        frame_id = [int(x[1]) for x in lines]
        pattern = [x[0] + "/{:06d}.JPEG" for x in lines]
        frame_seg_id = [int(x[2]) for x in lines]
        frame_seg_len = [int(x[3]) for x in lines]

    # cache keep infos dir
    cache_dir = os.path.dirname(img_index)
    keep = _filter_and_cache_keep_infos(dataset_name, image_set_index, anno_dir, cache_dir)

    if len(lines[0]) == 2:
        image_set_index = [image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
        frame_id = [frame_id[idx] for idx in range(len(keep)) if keep[idx]]
    else:
        image_set_index = [image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
        pattern = [pattern[idx] for idx in range(len(keep)) if keep[idx]]
        frame_id = [frame_id[idx] for idx in range(len(keep)) if keep[idx]]
        frame_seg_id = [frame_seg_id[idx] for idx in range(len(keep)) if keep[idx]]
        frame_seg_len = [frame_seg_len[idx] for idx in range(len(keep)) if keep[idx]]

    logger.info("Load {}'s annotations ...".format(dataset_name))
    # cache annotations
    cache_dir = os.path.dirname(img_index)
    dicts = _load_and_cache_annotations_mp(
        dataset_name,
        img_dir,
        anno_dir,
        image_set_index,
        frame_id,
        pattern,
        frame_seg_id,
        frame_seg_len,
        cache_dir)
    return dicts


if __name__ == "__main__":
    """
    Test the ILSVRC dataset loader.

    Usage:
        python -m detectron2.data.datasets.ilsvrc
    """
    import cv2
    from PIL import Image
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data.datasets.builtin import _PREDEFINED_SPLITS_IMAGENET
    logger = setup_logger(name=__name__)
    meta = get_imagenet_vid_instances_meta()

    root = "/data/datasets/"
    name = "imagenet_vid_train_15frames"
    img_dir, ann_dir, image_index = _PREDEFINED_SPLITS_IMAGENET[name]
    img_dir = root + img_dir
    ann_dir = root + ann_dir
    image_index = root + image_index

    dicts = load_imagenet_vid_instances(name, img_dir, ann_dir, image_index)

    num_samples = 3
    for sample in dicts[: num_samples]:
        img = np.array(Image.open(sample["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(sample)
        cv2.imshow(sample["image_id"], vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

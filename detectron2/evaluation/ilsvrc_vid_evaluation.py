# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/detectron2/evaluation/ilsvrc_vid_evaluation.py
@Time         : 2020-11-28 16:27:23
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-12-01 22:11:39
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

import copy
import itertools
import logging
import os
import pickle
import numpy as np
from tabulate import tabulate
from collections import OrderedDict, defaultdict
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import MetadataCatalog
from detectron2.utils.logger import create_small_table
from detectron2.evaluation import DatasetEvaluator


class ILSVRCVIDEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
        """
        meta = MetadataCatalog.get(dataset_name)
        # "ILSVRC2015/Annotations/VID",
        self._anno_file_template = os.path.join(meta.anno_dir, "{}.xml")
        # "ILSVRC2015/ImageSets/VID_val_videos.txt",
        self._image_set_path = meta.img_index.replace("videos", "frames")
        self._thing_dataset_id_to_contiguous_id = meta.thing_dataset_id_to_contiguous_id
        self._class_names = meta.thing_classes
        self._anno_cache_path = os.path.join(os.path.dirname(meta.img_index), "eval_ann_cache.pkl")
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            frame_id = input["frame_id"]
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.tolist()
                classes = instances.pred_classes.tolist()
                for box, score, cls in zip(boxes, scores, classes):
                    xmin, ymin, xmax, ymax = box
                    self._predictions.append(
                        f"{frame_id} {cls} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                    )

    def evaluate(self):
        predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[ILSVRCVIDEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "ilsvrc_vid_instances_predictions.txt")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write("\n".join(predictions))

        return self._eval_bbox(self._predictions)

    def _eval_bbox(self, det_res):
        """
        Args:
            det_res (list or str):
              * list: like self._predictions
              * str: /path/to/ilsvrc_vid_instances_predictions.txt

        """
        if isinstance(det_res, str):
            assert PathManager.exists(det_res), det_res
            with PathManager.open(det_res) as f:
                predictions = f.readlines()
        elif isinstance(det_res, list):
            predictions = det_res
        else:
            raise RuntimeError("Unknow 'det_res' type: {}.".format(type(det_res)))

        aps = defaultdict(list)  # iou -> ap per class
        for thresh in range(50, 100, 5):
            ap = vid_eval(
                predictions,
                self._anno_file_template,
                self._image_set_path,
                list(self._thing_dataset_id_to_contiguous_id.keys()),
                self._anno_cache_path,
                thresh / 100.0,
            )
            aps[thresh].append(ap * 100)

        iou_type = "bbox"
        results = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        results["bbox"] = {
            "AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]
        }
        small_table = create_small_table(results["bbox"])
        self._logger.info("Evaluation results for {}: \n".format(iou_type) + small_table)

        mAP_per_cls = []
        for idx, name in enumerate(self._class_names):
            mAP_per_cls.append(
                ("{}".format(name), np.mean([v[0][idx] for v in aps.values()]))
            )

        # tabulate it
        N_COLS = min(6, len(mAP_per_cls) * 2)
        results_flatten = list(itertools.chain(*mAP_per_cls))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        dump_info = {
            "task": "bbox",
            "tables": [small_table, table],
        }
        _dump_to_markdown([dump_info])

        # Copy so the caller can do whatever with results
        return copy.deepcopy(results)


def _dump_to_markdown(dump_infos, md_file="README.md"):
    """
    Dump a Markdown file that records the model evaluation metrics and corresponding scores
    to the current working directory.
    Args:
        dump_infos (list[dict]): dump information for each task.
        md_file (str): markdown file path.
    """
    title = os.getcwd().split("/")[-1]
    with open(md_file, "w") as f:
        f.write("# {}  ".format(title))
        for dump_info_per_task in dump_infos:
            task_name = dump_info_per_task["task"]
            tables = dump_info_per_task["tables"]
            tables = [table.replace("\n", "  \n") for table in tables]
            f.write("\n\n## Evaluation results for {}:  \n\n".format(task_name))
            f.write(tables[0])
            f.write("\n\n### Per-category {} AP:  \n\n".format(task_name))
            f.write(tables[1])
            f.write("\n")


##############################################################################
#
# Below code is modified from
# https://github.com/msracver/Deep-Feature-Flow/blob/master/lib/dataset/imagenet_vid_eval.py
# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xizhou Zhu
# --------------------------------------------------------

"""
given a imagenet vid imdb, compute mAP
"""


def parse_vid_rec(filename, classhash, img_ids, defaultIOUthr=0.5, pixelTolerance=10):
    """
    parse imagenet vid record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['label'] = classhash[obj.find('name').text]
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [float(bbox.find('xmin').text),
                            float(bbox.find('ymin').text),
                            float(bbox.find('xmax').text),
                            float(bbox.find('ymax').text)]
        gt_w = obj_dict['bbox'][2] - obj_dict['bbox'][0] + 1
        gt_h = obj_dict['bbox'][3] - obj_dict['bbox'][1] + 1
        thr = (gt_w * gt_h) / ((gt_w + pixelTolerance) * (gt_h + pixelTolerance))
        obj_dict['thr'] = np.min([thr, defaultIOUthr])
        objects.append(obj_dict)
    return {'bbox': np.array([x['bbox'] for x in objects]),
            'label': np.array([x['label'] for x in objects]),
            'thr': np.array([x['thr'] for x in objects]),
            'img_ids': img_ids}


def vid_ap(rec, prec):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :return: average precision
    """

    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def vid_eval(det_res, annopath, imageset_file, classname, annocache, ovthresh=0.5):
    """
    imagenet vid evaluation
    :param det_res: a list contains detection results, each elem is a string, like:
        'img_id cls score xmin ymin xmax ymax'
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param annocache: caching annotations
    :param ovthresh: overlap threshold
    :return: rec, prec, ap
    """
    with open(imageset_file, 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    img_basenames = [x[0] for x in lines]
    gt_img_ids = [int(x[1]) for x in lines]
    classhash = dict(zip(classname, range(0, len(classname))))

    # load annotations from cache
    if not os.path.isfile(annocache):
        recs = []
        for ind, image_filename in enumerate(img_basenames):
            recs.append(
                parse_vid_rec(
                    annopath.format(image_filename),
                    classhash,
                    gt_img_ids[ind],
                    ovthresh
                )
            )
            if ind % 100 == 0:
                print('reading annotations for {:d}/{:d}'.format(ind
                                                                 + 1, len(img_basenames)), end="")
        print('saving annotations cache to {:s}'.format(annocache))
        with open(annocache, 'wb') as f:
            pickle.dump(recs, f)
    else:
        with open(annocache, 'rb') as f:
            recs = pickle.load(f)

    # extract objects in :param classname:
    npos = np.zeros(len(classname))
    for rec in recs:
        rec_labels = rec['label']
        for x in rec_labels:
            npos[x] += 1

    splitlines = [x.strip().split(' ') for x in det_res]
    img_ids = np.array([int(x[0]) for x in splitlines])
    obj_labels = np.array([int(x[1]) for x in splitlines])
    obj_confs = np.array([float(x[2]) for x in splitlines])
    obj_bboxes = np.array([[float(z) for z in x[3:]] for x in splitlines])

    # sort by confidence
    if obj_bboxes.shape[0] > 0:
        sorted_inds = np.argsort(img_ids)
        img_ids = img_ids[sorted_inds]
        obj_labels = obj_labels[sorted_inds]
        obj_confs = obj_confs[sorted_inds]
        obj_bboxes = obj_bboxes[sorted_inds, :]

    num_imgs = max(max(gt_img_ids), max(img_ids)) + 1
    obj_labels_cell = [None] * num_imgs
    obj_confs_cell = [None] * num_imgs
    obj_bboxes_cell = [None] * num_imgs
    start_i = 0
    id = img_ids[0]
    for i in range(0, len(img_ids)):
        if i == len(img_ids) - 1 or img_ids[i + 1] != id:
            conf = obj_confs[start_i:i + 1]
            label = obj_labels[start_i:i + 1]
            bbox = obj_bboxes[start_i:i + 1, :]
            sorted_inds = np.argsort(-conf)

            obj_labels_cell[id] = label[sorted_inds]
            obj_confs_cell[id] = conf[sorted_inds]
            obj_bboxes_cell[id] = bbox[sorted_inds, :]
            if i < len(img_ids) - 1:
                id = img_ids[i + 1]
                start_i = i + 1

    # go down detections and mark true positives and false positives
    tp_cell = [None] * num_imgs
    fp_cell = [None] * num_imgs

    for rec in recs:
        id = rec['img_ids']
        gt_labels = rec['label']
        gt_bboxes = rec['bbox']
        gt_thr = rec['thr']
        num_gt_obj = len(gt_labels)
        gt_detected = np.zeros(num_gt_obj)

        labels = obj_labels_cell[id]
        bboxes = obj_bboxes_cell[id]

        num_obj = 0 if labels is None else len(labels)
        tp = np.zeros(num_obj)
        fp = np.zeros(num_obj)

        for j in range(0, num_obj):
            bb = bboxes[j, :]
            ovmax = -1
            kmax = -1
            for k in range(0, num_gt_obj):
                if labels[j] != gt_labels[k]:
                    continue
                if gt_detected[k] > 0:
                    continue
                bbgt = gt_bboxes[k, :]
                bi = [np.max((bb[0], bbgt[0])), np.max((bb[1], bbgt[1])),
                      np.min((bb[2], bbgt[2])), np.min((bb[3], bbgt[3]))]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap as area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                        (bbgt[2] - bbgt[0] + 1.) * \
                        (bbgt[3] - bbgt[1] + 1.) - iw * ih
                    ov = iw * ih / ua
                    # makes sure that this object is detected according
                    # to its individual threshold
                    if ov >= gt_thr[k] and ov > ovmax:
                        ovmax = ov
                        kmax = k
            if kmax >= 0:
                tp[j] = 1
                gt_detected[kmax] = 1
            else:
                fp[j] = 1

        tp_cell[id] = tp
        fp_cell[id] = fp

    tp_all = np.concatenate([x for x in np.array(tp_cell)[gt_img_ids] if x is not None])
    fp_all = np.concatenate([x for x in np.array(fp_cell)[gt_img_ids] if x is not None])
    obj_labels = np.concatenate([x for x in np.array(obj_labels_cell)[gt_img_ids] if x is not None])
    confs = np.concatenate([x for x in np.array(obj_confs_cell)[gt_img_ids] if x is not None])

    sorted_inds = np.argsort(-confs)
    tp_all = tp_all[sorted_inds]
    fp_all = fp_all[sorted_inds]
    obj_labels = obj_labels[sorted_inds]

    ap = np.zeros(len(classname))
    for c in range(len(classname)):
        # compute precision recall
        fp = np.cumsum(fp_all[obj_labels == c])
        tp = np.cumsum(tp_all[obj_labels == c])
        rec = tp / float(npos[c])
        # avoid division by zero in case first detection matches a difficult ground ruth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap[c] = vid_ap(rec, prec)
    return ap

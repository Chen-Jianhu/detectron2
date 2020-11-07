# -*- encoding: utf-8 -*-
"""
@File          :   flying_chairs_evaluation.py
@Time          :   2020/06/20 7:14:52
@Author        :   Chen-Jianhu (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 10:28:05
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import json
import logging
import os
import torch

from collections import OrderedDict
from fvcore.common.file_io import PathManager
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.logger import create_small_table


class FlyingChairsEvaluator(DatasetEvaluator):
    """
    Evaluate optical flow prediction.
    """

    def __init__(self, dataset_name, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
        """
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._endpoint_errors = []
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image pairs and
                contains keys like "height", "width", "flow_map".
            outputs: the outputs of a model. It is either list of optical flow predictions
                (Tensor [2, H, W]) or list of dicts with key "flow" that contains optical flow
                prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            pred = output["flow"].to(self._cpu_device)
            gt = input["flow_map"]
            # Pixel level average
            epe = float(torch.norm(gt - pred, p=2, dim=-3).mean())
            self._endpoint_errors.append(epe)
            if self._output_dir is not None:
                record = {
                    "flow_map": input["flow_map_file"],
                    "pred": pred.numpy().tolist(),
                    "epe": epe
                }
                self._predictions.append(record)

    def evaluate(self):
        if self._distributed:
            synchronize()
            endpoint_errors = all_gather(self._endpoint_errors)
            endpoint_errors = [per_image for per_gpu in endpoint_errors for per_image in per_gpu]
            self._predictions = all_gather(self._predictions)
            if not is_main_process():
                return

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "flow_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        ave_epe = sum(endpoint_errors) / len(endpoint_errors)
        res = {"ave_epe": ave_epe}

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "flow_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

        results = OrderedDict({"flow": res})
        small_table = create_small_table(res)
        self._logger.info("Evaluation results for flow: \n" + small_table)
        dump_info_one_task = {
            "task": "flow",
            "tables": [small_table],
        }
        _dump_to_markdown([dump_info_one_task])
        return results


def _dump_to_markdown(dump_infos, md_file="README.md"):
    """
    Dump a Markdown file that records the model evaluation metrics and corresponding scores
    to the current working directory.

    Args:
        dump_infos (list[dict[task->metrics]]): dump information for each task.
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
            f.write("\n")

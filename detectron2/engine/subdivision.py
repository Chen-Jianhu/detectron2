# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/detectron2/engine/subdivision.py
@Time         : 2020-11-24 17:43:21
@Author       : Chen-Jianhu (jhchen.mail@gmail.com)
@Last Modified: 2020-11-24 23:27:38
@License      : Copyright(C), USTC
@Desc         : None
"""

import time
import torch

from .defaults import DefaultTrainer
from .train_loop import _nullcontext

from detectron2.utils import comm


class BatchSubdivisionTrainer(DefaultTrainer):
    """
    Usage same as DefaultTrainer.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        # For simulate large batch training
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        batch_subdivisions = cfg.SOLVER.BATCH_SUBDIVISIONS

        assert (
            batch_subdivisions > 0
        ), "cfg.SOLVER.BATCH_SUBDIVISIONS ({}) must be greater than or equal to 1.".format(
            batch_subdivisions
        )

        if batch_subdivisions > 1:
            # if batch_subdivisions is equal to 1, the following check is redundant
            assert (
                images_per_batch % batch_subdivisions == 0
            ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the " \
                "cfg.SOLVER.BATCH_SUBDIVISIONS ({}).".format(images_per_batch, batch_subdivisions)
            images_per_mini_batch = images_per_batch // batch_subdivisions

            num_workers = comm.get_world_size()
            assert (
                images_per_mini_batch % num_workers == 0
            ), "images per mini batch ({}, is calculated by cfg.SOLVER.IMS_PER_BATCH // " \
                "cfg.SOLVER.BATCH_SUBDIVISIONS) must be divisible by the number of workers " \
                "({}).".format(images_per_mini_batch, num_workers)

            assert (
                images_per_mini_batch >= num_workers
            ), "images per mini batch ({}, is calculated from cfg.SOLVER.IMS_PER_BATCH // " \
                "cfg.SOLVER.BATCH_SUBDIVISIONS) must be larger than the number of workers " \
                "({}).".format(images_per_mini_batch, num_workers)
        elif batch_subdivisions == 1:
            images_per_mini_batch = images_per_batch

        self.batch_subdivisions = batch_subdivisions
        cfg = BatchSubdivisionTrainer.scale_batchsize(cfg, images_per_mini_batch)

        super().__init__(cfg)

    @staticmethod
    def scale_batchsize(cfg, images_per_mini_batch):
        """
        Adjust batchsize according to SOLVER.BATCH_SUBDIVISIONS.
        """
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        cfg.SOLVER.IMS_PER_BATCH = images_per_mini_batch

        if frozen:
            cfg.freeze()
        return cfg

    def run_step(self):
        """
        Implement the batch subdivision training logic.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        sum_data_time = 0.

        self.optimizer.zero_grad()

        for _ in range(self.batch_subdivisions):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            data_time = time.perf_counter() - start
            sum_data_time += data_time

            loss_dict = self.model(data)
            losses = sum(loss_dict.values())

            losses.backward()

        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(
            torch.cuda.Stream()
        ) if losses.device.type == "cuda" else _nullcontext():
            metrics_dict = loss_dict
            metrics_dict["data_time"] = sum_data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        self.optimizer.step()

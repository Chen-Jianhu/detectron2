# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/detectron2/engine/subdivision.py
@Time         : 2020-11-25 23:24:28
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-27 22:13:11
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

import time

from detectron2.utils import comm
from .defaults import DefaultTrainer


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
        self._trainer.iter = self.iter

        model = self._trainer.model
        optimizer = self._trainer.optimizer
        _data_loader_iter = self._trainer._data_loader_iter
        _write_metrics = self._trainer._write_metrics

        assert model.training, "[SimpleTrainer] model was changed to eval mode!"

        optimizer.zero_grad()

        sum_data_time = 0.
        for _ in range(self.batch_subdivisions):
            start = time.perf_counter()
            data = next(_data_loader_iter)
            data_time = time.perf_counter() - start
            sum_data_time += data_time

            loss_dict = model(data)
            losses = sum(loss_dict.values())

            losses.backward()

        _write_metrics(loss_dict, sum_data_time)

        optimizer.step()

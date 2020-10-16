# -*- encoding: utf-8 -*-
"""
@File          :   defaults.py
@Time          :   2020/07/02 22:53:06
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/02 22:54:11
@License       :   Copyright(C), USTC
@Desc          :

"""
import time
import torch

from .defaults import DefaultTrainer
from .train_loop import _nullcontext

from detectron2.utils import comm


class BatchSubdivisionTrainer(DefaultTrainer):
    """
    A trainer with default training logic.
    It is a subclass of :class:`SimpleTrainer` and instantiates everything needed from the
    config. It does the following:

    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
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
        Implement the standard training logic described above.
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

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

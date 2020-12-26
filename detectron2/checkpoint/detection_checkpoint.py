# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/detectron2/checkpoint/detection_checkpoint.py
@Time         : 2020-11-27 22:19:44
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-30 22:15:09
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

# Copyright (c) Facebook, Inc. and its affiliates.
import pickle
from fvcore.common.checkpoint import Checkpointer

import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager

from .c2_model_loading import align_and_update_state_dicts


def _fliter_out_key_startswith(state_dict, starts):
    """
    Args：
        starts (str or list):
    """
    assert isinstance(starts, (str, list))
    if isinstance(starts, str):
        starts = [starts]

    remove_keys = []
    for s in starts:
        for name in state_dict.keys():
            if name.startswith(s):
                remove_keys.append(name)

    for key in remove_keys:
        state_dict.pop(key)


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        if hasattr(self, "path_manager"):
            self.path_manager = PathManager
        else:
            # This could only happen for open source
            # TODO remove after upgrading fvcore
            from fvcore.common.file_io import PathManager as g_PathManager

            for handler in PathManager._path_handlers.values():
                try:
                    g_PathManager.register_handler(handler)
                except KeyError:
                    pass

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()

            if "flow_div" in checkpoint["model"].keys():
                _fliter_out_key_startswith(model_state_dict, ["backbone", "roi_heads"])
            else:
                _fliter_out_key_startswith(model_state_dict, "flow_net")

            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)
        if incompatible is None:  # support older versions of fvcore
            return None

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible

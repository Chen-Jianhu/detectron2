# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/projects/SSD/ssd/config/config.py
@Time         : 2020-11-24 17:43:21
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-24 23:32:29
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

from detectron2.config import CfgNode as CN


def add_ssd_config(cfg):
    """
    Add config for SSD.
    """
    cfg.MODEL.META_ARCHITECTURE = "SSD"
    cfg.MODEL.BACKBONE.NAME = "build_ssd_vgg_backbone"

    # VGG config
    cfg.MODEL.VGG = CN()
    cfg.MODEL.VGG.ARCH = "D"
    cfg.MODEL.VGG.OUT_FEATURES = ["conv4_3", "conv7"]
    cfg.MODEL.VGG.POOL_ARGS = CN()
    cfg.MODEL.VGG.POOL_ARGS.POOL3 = [2, 2, 0, True]  # k, s, p, ceil_cfg.MODEL
    cfg.MODEL.VGG.POOL_ARGS.POOL4 = [3, 1, 1, False]  # k, s, p, ceil_cfg.MODEL
    cfg.MODEL.VGG.FC_TO_CONV = True

    # SSD config
    cfg.MODEL.SSD = CN()

    cfg.MODEL.SSD.NUM_CLASSES = 80
    cfg.MODEL.SSD.IN_FEATURES = ["conv4_3", "conv7"]
    cfg.MODEL.SSD.EXTRA_LAYER_ARCH = CN()
    cfg.MODEL.SSD.EXTRA_LAYER_ARCH.SIZE300 = [
        256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    cfg.MODEL.SSD.EXTRA_LAYER_ARCH.SIZE512 = [
        256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 256],
    cfg.MODEL.SSD.IOU_THRESHOLDS = [0.5, 0.5]
    cfg.MODEL.SSD.IOU_LABELS = [0, -1, 1]
    cfg.MODEL.SSD.BBOX_REG_WEIGHTS = [10.0, 10.0, 5.0, 5.0]
    cfg.MODEL.SSD.L2NORM_SCALE = 20.0
    # Loss parameters:
    cfg.MODEL.SSD.LOSS_ALPHA = 1.0
    cfg.MODEL.SSD.SMOOTH_L1_LOSS_BETA = 1.0
    cfg.MODEL.SSD.NEGATIVE_POSITIVE_RATIO = 3.0
    # Inference parameters:
    cfg.MODEL.SSD.SCORE_THRESH_TEST = 0.02
    cfg.MODEL.SSD.NMS_THRESH_TEST = 0.45

    # default is 300 size
    cfg.MODEL.SSD.IMAGE_SIZE = 300
    cfg.MODEL.SSD.FEATURE_MAP_SIZE = [38, 19, 10, 5, 3, 1]

    # Anchor config
    cfg.MODEL.SSD.DEFAULT_BOX = CN()

    cfg.MODEL.SSD.DEFAULT_BOX.SCALE = CN()
    cfg.MODEL.SSD.DEFAULT_BOX.SCALE.CONV4_3_SCALE = 0.07
    cfg.MODEL.SSD.DEFAULT_BOX.SCALE.S_MIN = 0.15
    cfg.MODEL.SSD.DEFAULT_BOX.SCALE.S_MAX = 0.9

    cfg.MODEL.SSD.DEFAULT_BOX.SIZES = [[44, 60], [88, 120],
                                       [176, 240], [352, 480],
                                       [704, 960], [1408, 1920]]
    cfg.MODEL.SSD.DEFAULT_BOX.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    cfg.MODEL.SSD.DEFAULT_BOX.CLIP = False

    # Augmentations config
    cfg.INPUT.COLOR_AUG = True
    cfg.INPUT.SWAP_CHANNELS = True
    cfg.INPUT.EXPAND = True
    cfg.INPUT.MIN_IOU_CROP = True
    cfg.INPUT.RESIZE = CN({"ENABLED": True})
    cfg.INPUT.RESIZE.SIZE = [300, 300]

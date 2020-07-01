# -*- encoding: utf-8 -*-
"""
@File          :   roi_heads.py
@Time          :   2020/06/22 21:45:47
@Author        :   Facebook, Inc. and its affiliates.
@Modified By   :   Jianhu Chen (jhchen.mail@gmail.com)
@Last Modified :   2020/07/01 23:00:03
@License       :   Copyright(C), USTC
@Desc          :   None
"""

import torch
import torch.nn as nn

from fvcore.nn import weight_init
from detectron2.layers import ShapeSpec, Conv2d
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    ROIHeads,
    build_mask_head,
    select_foreground_proposals,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.modeling.box_regression import Box2BoxTransform
from rfcn.modeling.poolers import ROIPooler


@ROI_HEADS_REGISTRY.register()
class RFCNROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        # fmt: off
        self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        new_conv_channels = cfg.MODEL.ROI_HEADS.NEW_CONV_CHANNELS
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        self.smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.box_reg_loss_type = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE
        self.box_reg_loss_weight = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(self.in_features) == 1

        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        self.new_conv_layer = Conv2d(
            input_shape[self.in_features[0]].channels,
            new_conv_channels,
            kernel_size=1,
            activation=nn.ReLU(inplace=True))

        self.rfcn_cls = Conv2d(
            new_conv_channels,
            (self.num_classes + 1) * (pooler_resolution ** 2),
            kernel_size=1)
        self.rfcn_bbox = Conv2d(
            new_conv_channels,
            4 * (pooler_resolution ** 2),
            kernel_size=1)

        self.psroipooled_cls_rois = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.psroipooled_loc_rois = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.ave_cls_score_rois = nn.AvgPool2d(pooler_resolution)
        self.ave_bbox_pred_rois = nn.AvgPool2d(pooler_resolution)

        for layer in [self.new_conv_layer, self.rfcn_cls, self.rfcn_bbox]:
            weight_init.c2_msra_fill(layer)

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(
                    channels=new_conv_channels,
                    width=pooler_resolution,
                    height=pooler_resolution
                ),
            )

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]

        features = features[self.in_features[0]]  # res5
        features = self.new_conv_layer(features)  # reducing dimension

        ps_score_map_cls = self.rfcn_cls(features)
        ps_score_map_bbox = self.rfcn_bbox(features)

        cls_features = self.psroipooled_cls_rois([ps_score_map_cls], proposal_boxes)
        box_features = self.psroipooled_cls_rois([ps_score_map_bbox], proposal_boxes)

        # shape (N, C+1), scores for each of the N box. Each row contains the scores for
        # C object categories and 1 background class.
        cls_preds = self.ave_cls_score_rois(cls_features)[:, :, 0, 0]
        # bounding box regression deltas for each box. Shape is shape (N, Cx4), or (N,4)
        # for class-agnostic regression.
        bbox_preds = self.ave_bbox_pred_rois(box_features)[:, :, 0, 0]

        if self.training:
            del features
            losses = FastRCNNOutputs(
                self.box2box_transform,
                cls_preds,
                bbox_preds,
                proposals,
                self.smooth_l1_beta,
                self.box_reg_loss_type,
                self.box_reg_loss_weight,
            ).losses()
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(bbox_preds, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances

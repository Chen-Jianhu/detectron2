# -*- encoding: utf-8 -*-
"""
@File         : /detectron2/projects/SSD/ssd/modeling/meta_arch/ssd.py
@Time         : 2020-11-24 17:43:21
@Author       : Facebook, Inc. and its affiliates.
@Last Modified: 2020-11-24 23:36:28
@Modified By  : Chen-Jianhu (jhchen.mail@gmail.com)
@License      : Copyright(C), USTC
@Desc         : None
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import List

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.layers import ShapeSpec, cat, batched_nms
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb

from ..default_box import DefaultBox


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


@META_ARCH_REGISTRY.register()
class SSD(nn.Module):
    """
    Implement SSD (https://arxiv.org/abs/1512.02325).
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg):
        super().__init__()

        self.image_size = cfg.MODEL.SSD.IMAGE_SIZE
        self.num_classes = cfg.MODEL.SSD.NUM_CLASSES
        self.in_features = cfg.MODEL.SSD.IN_FEATURES
        self.extra_layer_arch = cfg.MODEL.SSD.EXTRA_LAYER_ARCH["SIZE{}".format(self.image_size)]
        self.l2norm_scale = cfg.MODEL.SSD.L2NORM_SCALE
        # Loss parameters:
        self.loss_alpha = cfg.MODEL.SSD.LOSS_ALPHA
        self.smooth_l1_loss_beta = cfg.MODEL.SSD.SMOOTH_L1_LOSS_BETA
        self.negative_positive_ratio = cfg.MODEL.SSD.NEGATIVE_POSITIVE_RATIO
        # Inference parameters:
        self.score_threshold = cfg.MODEL.SSD.SCORE_THRESH_TEST
        self.nms_threshold = cfg.MODEL.SSD.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        # Build extra layers
        self.extra_layers = self._make_extra_layers(
            feature_shapes[-1].channels, self.extra_layer_arch)
        extra_layer_channels = [c for c in self.extra_layer_arch if isinstance(c, int)]
        feature_shapes += [ShapeSpec(channels=c) for c in extra_layer_channels[1::2]]

        # Head
        self.head = SSDHead(cfg, feature_shapes)
        self.l2norm = L2Norm(backbone_shape[self.in_features[0]].channels, self.l2norm_scale)
        self.default_box_generator = DefaultBox(cfg)
        self.default_boxes = self.default_box_generator()

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.SSD.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.SSD.IOU_THRESHOLDS,
            cfg.MODEL.SSD.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        # Initialization
        self._init_weights()

    def _init_weights(self):
        # extra layers param init
        for layer in self.extra_layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        # l2 norm param init
        for param in self.l2norm.parameters():
            torch.nn.init.constant_(param, self.l2norm_scale)

    @property
    def device(self):
        return self.pixel_mean.device

    def _make_extra_layers(self, in_channels, extra_arch):
        extra_layers = list()
        flag = False  # kernel size flag
        for idx, v in enumerate(extra_arch):
            if in_channels != 'S':
                if v == 'S':
                    extra_layers += [nn.Conv2d(in_channels, extra_arch[idx + 1],
                                               kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    extra_layers += [nn.Conv2d(in_channels,
                                               v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        if self.image_size == 512:
            extra_layers[-1] = nn.Conv2d(extra_arch[-2], extra_arch[-1], kernel_size=4, padding=1)

        return nn.ModuleList(extra_layers)

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        # Vis
        # from detectron2.utils.visualizer import fast_visualize_sample
        # sample = batched_inputs[0]
        # fast_visualize_sample(sample, "coco_2017_train")

        images = self.preprocess_image(batched_inputs)
        # vgg feature maps: ['conv4_3', 'conv7']
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        # featrue map: conv4_3
        # SSD :paper: conv4_3 has a different feature scale compared to the other layers, we use
        # the L2 normalization technique to scale the feature norm at each location
        # in the feature map to 20 and learn the scale during back propagation.
        features[0] = self.l2norm(features[0])

        # conv7
        x = features[-1]
        # compute featrue maps: conv8_2, conv9_2, conv10_2, and conv11_2
        for idx, extra_layer in enumerate(self.extra_layers):
            x = F.relu(extra_layer(x), inplace=True)
            if idx % 2 == 1:
                features.append(x)

        conf_pred, loc_pred = self.head(features)

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_conf, gt_default_boxes_deltas = self.label_anchors(self.default_boxes, gt_instances)
            losses = self.losses(gt_conf, gt_default_boxes_deltas, conf_pred, loc_pred)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(conf_pred, loc_pred, self.default_boxes, images)
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(conf_pred, loc_pred, self.default_boxes, images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, gt_conf, gt_default_boxes_deltas, conf_pred, loc_pred):
        """
        SSD Weighted Loss Function:
            L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
            Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
            weighted by α, which is set to 1 by cross val.
            where:
                c: class confidences,
                l: predicted boxes,
                g: ground truth boxes
                N: number of matched default boxes
            See: https://arxiv.org/pdf/1512.02325.pdf for more details.

        Args:
            For `gt_conf` and `gt_default_boxes_deltas` parameters, see
                :method:`get_ground_truth`.
                Their concatenated shapes are [N, R] and [N, R, 4] respectively, where the R
                is the total number of default box, i.e. sum(Hi x Wi x D) for all levels, the
                C is the total number of class, the D is the number of default box in each location.
            For `conf_pred` and `loc_pred`, see: method:`SSDHead.forward`.
                Their shapes are [N, R, C] and [N, R,, 4] respectively.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_conf" and "loss_loc".
        """
        # shape=[#batch_size, #default_boxes, #num_classes] and [#batch_size, #default_boxes, 4]
        conf_pred = cat(conf_pred, dim=1)
        loc_pred = cat(loc_pred, dim=1)

        # filter out the negative samples
        positive_mask = gt_conf < 80

        # the number of matched default box
        num_pos_samples = positive_mask.sum()

        loss_conf, loss_loc = multi_apply(
            self.loss_single,
            conf_pred,
            loc_pred,
            gt_conf,
            gt_default_boxes_deltas,
            num_total_samples=num_pos_samples
        )
        return {"loss_conf": sum(loss_conf), "loss_loc": sum(loss_loc)}

    def loss_single(self,
                    conf_pred_i,
                    loc_pred_i,
                    gt_conf_i,
                    gt_default_boxes_deltas_i,
                    num_total_samples):
        """
        Calculate the loss of a single image.

        Args:
            conf_pred_i (Tensor): see: method: `losses`.
            loc_pred_i (Tensor): see: method: `losses`.
            gt_conf_i (Tensor): see: method: `losses`.
            gt_default_boxes_deltas_i (Tensor): see: method: `losses`.
            Their shapes are [R, C], [R, 4], [R] and [R, 4] respectively.
            num_total_samples (int): the number of matched default box.
        """
        # confidence loss
        loss_conf_all = F.cross_entropy(conf_pred_i, gt_conf_i, reduction='none')
        pos_idxs = (gt_conf_i < self.num_classes).nonzero().view(-1)
        neg_idxs = (gt_conf_i == self.num_classes).nonzero().view(-1)

        num_pos_samples = pos_idxs.size(0)
        num_neg_samples = int(self.negative_positive_ratio * num_pos_samples)
        if num_neg_samples > neg_idxs.size(0):
            num_neg_samples = neg_idxs.size(0)
        topk_loss_conf_neg, _ = loss_conf_all[neg_idxs].topk(num_neg_samples)
        loss_conf_pos = loss_conf_all[pos_idxs].sum()
        loss_con_neg = topk_loss_conf_neg.sum()
        # confidence loss including positive and negative samples
        loss_conf = (loss_conf_pos + loss_con_neg) / num_total_samples

        # localization loss
        loss_loc = F.smooth_l1_loss(
            loc_pred_i, gt_default_boxes_deltas_i, reduction='none').sum(dim=-1)
        loss_loc = loss_loc[pos_idxs].sum() / num_total_samples

        return loss_conf, loss_loc

    @torch.no_grad()
    def label_anchors(self, default_boxes, gt_instances):
        """
        Args:
            default_boxes (list[Boxes]): a list of 'Boxes' elements.
                The Boxes contains default boxes of one image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            gt_conf (Tensor):
                An integer tensor of shape [N, R] storing ground-truth labels for each default box.
                R is the total number of default box, i.e. the sum of Hi x Wi x D for all levels.

                * Default box with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, C-1] range.
                * Default box whose IoU are below the background threshold are assigned
                the label "C".
                * Default box whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.

            gt_default_boxes_deltas (Tensor): Shape [N, R, 4].
                The last dimension represents ground-truth box2box transform targets
                (g^cx, g^cy, g^w, g^h)that map each default box to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding default box
                is labeled as foreground.
        """
        gt_conf = list()
        gt_default_boxes_deltas = list()
        # list[Tensor(R, 4)], one for each image
        default_boxes = Boxes.cat(default_boxes)

        # each Instances (for one image)
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, default_boxes)
            gt_matched_idxs, default_box_labels = self.matcher(match_quality_matrix)

            if len(gt_per_image) > 0:
                # ground truth box regression
                matched_gt_boxes_i = gt_per_image.gt_boxes[gt_matched_idxs]

                # meaningful only when the corresponding default box is labeled as foreground.
                gt_default_boxes_deltas_i = self.box2box_transform.get_deltas(
                    default_boxes.tensor, matched_gt_boxes_i.tensor
                )

                gt_conf_i = gt_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_conf_i[default_box_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_conf_i[default_box_labels == -1] = -1
            else:
                gt_default_boxes_deltas_i = torch.zeros_like(default_boxes.tensor)
                gt_conf_i = torch.zeros_like(gt_matched_idxs) + self.num_classes

            gt_conf.append(gt_conf_i)
            gt_default_boxes_deltas.append(gt_default_boxes_deltas_i)

        return torch.stack(gt_conf), torch.stack(gt_default_boxes_deltas)

    def inference(self, conf_pred, loc_pred, default_boxes, images):
        """
        Args:
            conf_pred, loc_pred: Same as the output of :meth:`SSDHead.forward`
                shape = [N, Hi x Wi x D, 4] and [N, Hi x Wi x D, C].
            default_boxes (list['Boxes']):  a list of 'Boxes' elements.
                The Boxes contains default boxes of one image on the specific feature level.
            images (ImageList): the input images.

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = list()
        for img_idx in range(len(conf_pred[0])):
            image_size = images.image_sizes[img_idx]
            conf_pred_per_image = [
                conf_pred_per_level[img_idx] for conf_pred_per_level in conf_pred
            ]
            loc_pred_per_image = [
                loc_pred_per_level[img_idx] for loc_pred_per_level in loc_pred
            ]
            results_per_image = self.inference_single_image(
                conf_pred_per_image, loc_pred_per_image, default_boxes,
                tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self,
                               conf_pred_per_image,
                               loc_pred_per_image,
                               default_boxes,
                               image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Args:
            conf_pred_per_image (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size [Hi x Wi x D, C].
            loc_pred_per_image (list[Tensor]): same shape as 'conf_pred_per_image' except
                that C becomes 4.
            default_boxes (list['Boxes']):  a list of 'Boxes' elements.
                The Boxes contains default boxes of one image on the specific feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        # predict confidence
        conf_pred = torch.cat(conf_pred_per_image, dim=0)  # [R, C]
        conf_pred = conf_pred.softmax(dim=1)

        # predict boxes
        loc_pred = torch.cat(loc_pred_per_image, dim=0)  # [R, 4]
        default_boxes = Boxes.cat(default_boxes)  # [R, 4]
        boxes_pred = self.box2box_transform.apply_deltas(
            loc_pred, default_boxes.tensor)

        num_boxes, num_classes = conf_pred.shape
        boxes_pred = boxes_pred.view(num_boxes, 1, 4).expand(
            num_boxes, num_classes, 4)  # [R, C, 4]
        labels = torch.arange(num_classes, device=self.device)  # [0, ..., C]
        labels = labels.view(1, num_classes).expand_as(conf_pred)  # [R, C]

        # remove predictions with the background label
        boxes_pred = boxes_pred[:, :-1]
        conf_pred = conf_pred[:, :-1]
        labels = labels[:, :-1]

        # batch everything, by making every class prediction be a separate instance
        boxes_pred = boxes_pred.reshape(-1, 4)
        conf_pred = conf_pred.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        indices = torch.nonzero(conf_pred > self.score_threshold).squeeze(1)
        boxes_pred, conf_pred, labels = boxes_pred[indices], conf_pred[indices], labels[indices]

        keep = batched_nms(boxes_pred, conf_pred, labels, self.nms_threshold)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_pred[keep])
        result.scores = conf_pred[keep]
        result.pred_classes = labels[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class SSDHead(nn.Module):
    """
    The head used in SSD for object classification and box regression.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        self.num_classes = cfg.MODEL.SSD.NUM_CLASSES
        self.default_box_aspect_ratios = cfg.MODEL.SSD.DEFAULT_BOX.ASPECT_RATIOS

        # build classification subnet and localization subnet
        # number of boxes per feature map location
        mbox = [(len(a_r) + 1) * 2 for a_r in self.default_box_aspect_ratios]
        self.cls_subnet = nn.ModuleList()
        self.bbox_subnet = nn.ModuleList()
        for i, m in zip(input_shape, mbox):
            self.cls_subnet.append(
                nn.Conv2d(i.channels, m * (self.num_classes + 1),
                          kernel_size=3, padding=1)
            )
            self.bbox_subnet.append(
                nn.Conv2d(i.channels, m * 4, kernel_size=3, padding=1)
            )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        for layer in [*self.cls_subnet, *self.bbox_subnet]:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): VGG16-D feature map tensors.
                We use conv4_3, conv7(fc7), conv8_2, conv9_2, conv10_2, and conv11_2 to predict
                both location and confidences.
        Returns:
            conf_pred (list[Tensor]): a list of tensors, each has shape (N, HWD, K).
                These tensors predicts the classification confidences of default box at each
                feature map.
            loc_pred (list[Tensor]): a list of tensors, each has shape (N, HWD, 4).
                The tensor predicts 4-vector (g^cx, g^cy, g^w, g^h) box regression values for
                every default box.
        """
        # compute confidences and location
        conf_pred = list()
        loc_pred = list()
        for feature, cls_module, bbox_module in zip(features, self.cls_subnet, self.bbox_subnet):
            # permute: conf_pred[i].shape from [N, C, Hi, Wi] to [N, Hi, Wi, C]
            conf_pred.append(cls_module(feature).permute(
                0, 2, 3, 1).contiguous())
            loc_pred.append(bbox_module(feature).permute(
                0, 2, 3, 1).contiguous())

        # resize to (N, HWD, 4) and (N, HWD, K).
        conf_pred = [
            result.view(result.size(0), -1, (self.num_classes + 1)) for result in conf_pred
        ]
        loc_pred = [
            result.view(result.size(0), -1, 4) for result in loc_pred
        ]

        return conf_pred, loc_pred


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float)
                * x_float / norm).type_as(x)

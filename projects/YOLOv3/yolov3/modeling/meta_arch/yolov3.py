from detectron2.layers import Conv2d
from detectron2.layers import get_norm
from detectron2.layers import CNNBlockBase
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import List

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.layers import ShapeSpec, cat, batched_nms
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb

from ..matcher import YOLOMatcher


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


@META_ARCH_REGISTRY.register()
class YOLOv3(nn.Module):
    """
    Implement YOLOV3 (https://arxiv.org/abs/1512.02325).
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.MODEL.YOLOV3.NUM_CLASSES
        self.in_features = cfg.MODEL.YOLOV3.IN_FEATURES
        # Loss parameters:
        self.loss_alpha = cfg.MODEL.YOLOV3.LOSS_ALPHA
        self.smooth_l1_loss_beta = cfg.MODEL.YOLOV3.SMOOTH_L1_LOSS_BETA
        self.negative_positive_ratio = cfg.MODEL.YOLOV3.NEGATIVE_POSITIVE_RATIO
        # Inference parameters:
        self.score_threshold = cfg.MODEL.YOLOV3.SCORE_THRESH_TEST
        self.nms_threshold = cfg.MODEL.YOLOV3.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        self.head = YOLOV3Head(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.YOLOV3.BBOX_REG_WEIGHTS)
        self.matcher = YOLOMatcher(
            cfg.MODEL.YOLOV3.IOU_THRESHOLDS,
            cfg.MODEL.YOLOV3.IOU_LABELS,
        )

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

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
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        # `predictions` is a list contains #levels tensor
        # each element shape is [#N, #H, #w, #A * (#C + 5)]
        predictions = self.head(features)

        # TODO fix this logic
        conf_pred, loc_pred = predictions

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
        YOLOV3 Weighted Loss Function:
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
            For `conf_pred` and `loc_pred`, see: method:`YOLOV3Head.forward`.
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
            conf_pred, loc_pred: Same as the output of :meth:`YOLOV3Head.forward`
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


class YOLOV3Head(nn.Module):
    """
    The head used in YOLOV3 for object classification and box regression.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        self.num_classes = cfg.MODEL.YOLOV3.NUM_CLASSES
        num_anchors_per_level = build_anchor_generator(cfg, input_shape).num_cell_anchors
        norm = cfg.MODEL.YOLOV3.NORM

        assert len(num_anchors_per_level) == len(input_shape), (
            "The number of anchor types is {}, but the number of input features is {}.".format(
                len(num_anchors_per_level), len(input_shape)
            )
        )

        # Reverse input_shape and anchors
        # feature maps order change to ["stage5", "stage4", "stage3"]
        input_shape = input_shape[::-1]

        yolo_blocks = []
        up_sample_convs = []
        pred_nets = []
        for idx in range(len(input_shape)):
            if idx == 0:
                in_channels = input_shape[idx].channels
            else:
                in_channels = input_shape[idx].channels + input_shape[idx - 1].channels // 4
            out_channels = input_shape[idx].channels // 2
            block_i = YOLOBlock(in_channels, out_channels, norm=norm)
            yolo_blocks.append(block_i)

            if idx != len(input_shape) - 1:
                up_sample_conv_i = Conv2d(
                    out_channels,
                    out_channels // 2,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    norm=get_norm(norm, out_channels // 2),
                    activation=nn.LeakyReLU(0.1, inplace=True)
                )
                up_sample_convs.append(up_sample_conv_i)

            pred_channels = num_anchors_per_level[idx] * (self.num_classes + 5)
            pred_net_i = [
                Conv2d(
                    out_channels,
                    out_channels * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm=get_norm(norm, out_channels * 2),
                    activation=nn.LeakyReLU(0.1, inplace=True)
                ),
                Conv2d(
                    out_channels * 2,
                    pred_channels,
                    kernel_size=1,
                    stride=1,
                ),
            ]
            pred_nets.append(nn.Sequential(*pred_net_i))

        self.yolo_blocks = nn.ModuleList(yolo_blocks)
        self.up_sample_convs = nn.ModuleList(up_sample_convs)
        self.pred_nets = nn.ModuleList(pred_nets)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        # TODO
        pass

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
        # Same as above, reverse feature maps
        features = features[::-1]

        predictions = []
        block_features_i = None
        for idx in range(len(features)):
            features_i = features[idx]

            if idx != 0:
                features_i = cat([block_features_i, features_i], dim=1)

            block_features_i = self.yolo_blocks[idx](features_i)

            pred_i = self.pred_nets[idx](block_features_i)
            # permute: pred_i.shape from [N, C, Hi, Wi] to [N, Hi, Wi, C]
            pred_i = pred_i.permute(0, 2, 3, 1).contiguous()
            predictions.append(pred_i)

            # up sample
            if idx != len(features) - 1:
                block_features_i = self.up_sample_convs[idx](block_features_i)
                block_features_i = F.interpolate(block_features_i, scale_factor=2)

        return predictions


class YOLOBlock(CNNBlockBase):

    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        """
        Args:
            in_channels (int):
            stride (int):
            norm (str):
        """
        super().__init__(in_channels, out_channels, stride)

        convs = [
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, out_channels),
                activation=nn.LeakyReLU(0.1, inplace=True)
            ),
            Conv2d(
                out_channels,
                out_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=get_norm(norm, out_channels * 2),
                activation=nn.LeakyReLU(0.1, inplace=True)
            ),
            Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, out_channels),
                activation=nn.LeakyReLU(0.1, inplace=True)
            ),
            Conv2d(
                out_channels,
                out_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=get_norm(norm, out_channels * 2),
                activation=nn.LeakyReLU(0.1, inplace=True)
            ),
            Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, out_channels),
                activation=nn.LeakyReLU(0.1, inplace=True)
            ),
        ]

        self.convs = nn.Sequential(*convs)

        # TODO
        # for m in self.modules():
        #     weight_init.c2_msra_fill(m)

    def forward(self, x):
        out = self.convs(x)
        return out

import math
import torch

from itertools import product as product
from detectron2.structures import Boxes


class DefaultBox:
    """Compute default box coordinates (xmin, ymin, xmax, ymax) for each feature map.
    Returns:
        default_boxes (list['Boxes']): a list of 'Boxes' elements.
            The Boxes contains default box of this image on the specific feature level.
    """

    def __init__(self, cfg):
        super(DefaultBox, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.feature_map_size = cfg.MODEL.SSD.FEATURE_MAP_SIZE
        self.image_size = cfg.MODEL.SSD.IMAGE_SIZE
        self.conv4_3_scale = cfg.MODEL.SSD.DEFAULT_BOX.SCALE.CONV4_3_SCALE
        self.s_min = cfg.MODEL.SSD.DEFAULT_BOX.SCALE.S_MIN
        self.s_max = cfg.MODEL.SSD.DEFAULT_BOX.SCALE.S_MAX
        self.aspect_ratios = cfg.MODEL.SSD.DEFAULT_BOX.ASPECT_RATIOS
        self.clip = cfg.MODEL.SSD.DEFAULT_BOX.CLIP

    def __call__(self):
        # compute box_size
        m = len(self.feature_map_size) - 1
        size_stride = math.floor(
            (math.floor(self.s_max * 100) - math.floor(self.s_min * 100)
             ) / (m - 1))
        bbox_size = [self.conv4_3_scale * self.image_size]
        bbox_size += [
            (self.s_min + i * size_stride / 100) * self.image_size
            for i in range(m)
        ]
        bbox_size += [1.05 * self.image_size]

        self.widths = [[] for _ in self.aspect_ratios]
        self.heights = [[] for _ in self.aspect_ratios]

        # each a_r denotes the aspect ratios of one feature map
        for i, a_rs in enumerate(self.aspect_ratios):
            # ratio = 1
            a_r = 1
            self.widths[i].append(bbox_size[i] * math.sqrt(a_r))
            self.heights[i].append(bbox_size[i] / math.sqrt(a_r))
            self.widths[i].append(
                math.sqrt(bbox_size[i] * bbox_size[i + 1]) * math.sqrt(a_r))
            self.heights[i].append(
                math.sqrt(bbox_size[i] * bbox_size[i + 1]) / math.sqrt(a_r))

            # other ratios
            for a_r in a_rs:
                self.widths[i].append(bbox_size[i] * math.sqrt(a_r))
                self.heights[i].append(bbox_size[i] / math.sqrt(a_r))
                a_r = 1 / a_r
                self.widths[i].append(bbox_size[i] * math.sqrt(a_r))
                self.heights[i].append(bbox_size[i] / math.sqrt(a_r))

        # compute center of default boxes
        self.center_xs = [[] for _ in self.feature_map_size]
        self.center_ys = [[] for _ in self.feature_map_size]
        for k, f_k in enumerate(self.feature_map_size):
            for i, j in product(range(f_k), repeat=2):
                # bbox center x, y
                cx = (j + 0.5) / f_k * self.image_size
                cy = (i + 0.5) / f_k * self.image_size
                self.center_xs[k].append(cx)
                self.center_ys[k].append(cy)

        default_boxes = []
        for i, cxs, cys in zip(range(len(self.feature_map_size)), self.center_xs, self.center_ys):
            one_feature_map_boxes = []
            widths = self.widths[i]
            heights = self.heights[i]
            for cx, cy in zip(cxs, cys):
                for w, h in zip(widths, heights):
                    (xmin, ymin, xmax, ymax) = cx - 0.5 * \
                        w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h
                    one_feature_map_boxes.append([xmin, ymin, xmax, ymax])
            one_feature_map_boxes = torch.tensor(one_feature_map_boxes, device=self.device)
            if self.clip:
                one_feature_map_boxes = one_feature_map_boxes.clamp_(
                    max=self.image_size, min=0)
            default_boxes.append(Boxes(one_feature_map_boxes))
        return default_boxes

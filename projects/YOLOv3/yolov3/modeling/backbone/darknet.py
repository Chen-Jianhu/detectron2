import torch
import numpy as np
import fvcore.nn.weight_init as weight_init

from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm, CNNBlockBase
from detectron2.modeling import Backbone, BACKBONE_REGISTRY

__all__ = [
    "BasicBlock",
    "BasicStem",
    "Darknet",
    "build_darknet_backbone",
]


class BasicStem(CNNBlockBase):

    def __init__(self, in_channels=3, out_channels=32, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride=1)
        self.in_channels = in_channels
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
            activation=nn.LeakyReLU(0.1, inplace=True)
        )

        weight_init.c2_msra_fill(self.conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class BasicBlock(CNNBlockBase):

    def __init__(self, in_channels, *, is_first_block=False, norm="BN"):
        """
        Args:
            in_channels (int):
            stride (int):
            norm (str):
        """
        out_channels = in_channels * 2
        stride = 2 if is_first_block else 1
        super().__init__(in_channels, out_channels, stride)

        if is_first_block:
            self.down_sample = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm=get_norm(norm, out_channels),
                activation=nn.LeakyReLU(0.1, inplace=True)
            )
            weight_init.c2_msra_fill(self.down_sample)

        self.conv1 = Conv2d(
            out_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, in_channels),
            activation=nn.LeakyReLU(0.1, inplace=True)
        )

        self.conv2 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
            activation=nn.LeakyReLU(0.1, inplace=True)
        )

        for layer in [self.conv1, self.conv2]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        if hasattr(self, "down_sample"):
            x = self.down_sample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        return out


class Darknet(Backbone):
    """
    Implement :paper:`Darknet`.
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "stage" + str(i + 1)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]
        """
        assert x.dim() == 4, f"Darknet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, (stage, _) in enumerate(self.stages_and_names, start=1):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(num_blocks, in_channels, norm="BN"):
        """
        Create a list of blocks of the same type that forms one ResNet stage.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            first_stride (int): deprecated
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.

        Returns:
            list[nn.Module]: a list of block module.

        Examples:
        ::
            stages = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). In this case ``stride_per_block[1:]`` should all be 1.
        """
        blocks = []
        for idx in range(num_blocks):
            is_first_block = (idx == 0)
            block_i = BasicBlock(in_channels, is_first_block=is_first_block, norm=norm)
            blocks.append(block_i)
        return blocks


@BACKBONE_REGISTRY.register()
def build_darknet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.DARKNET.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.DARKNET.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.DARKNET.OUT_FEATURES
    depth = cfg.MODEL.DARKNET.DEPTH
    in_channels = cfg.MODEL.DARKNET.STEM_OUT_CHANNELS
    # fmt: on

    num_blocks_per_stage = {
        53: [1, 2, 8, 8, 4],
    }[depth]

    stages = []

    out_stage_idx = [{"stage3": 3, "stage4": 4, "stage5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx in range(max_stage_idx):
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "in_channels": in_channels,
            "norm": norm,
        }
        blocks = Darknet.make_stage(**stage_kargs)
        in_channels *= 2
        stages.append(blocks)
    return Darknet(stem, stages, out_features=out_features).freeze(freeze_at)

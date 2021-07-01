import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from lib.models.tools.module_helper import ModuleHelper
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.utils.tools.logger import Logger as Log
import numpy as np


class FPN(nn.Module):
    def __init__(self, in_channels,
                 out_channels=256,
                 num_outs=4,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(
                in_channels[i],
                out_channels,
                1)
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class SemanticFPNDecoder(nn.Module):
    def __init__(self, feature_strides, num_classes):
        super(SemanticFPNDecoder, self).__init__()
        self.in_channels = [64, 128, 320, 512]
        self.feature_strides = feature_strides
        self.scale_heads = nn.ModuleList()
        self.channels = 128
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    nn.Conv2d(
                        256 if k == 0 else self.channels,
                        self.channels,
                        kernel_size=3,
                        padding=1))
                scale_head.append(ModuleHelper.BNReLU(self.channels, bn_type='torchsyncbn'))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.cls_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, x):
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            output = output + nn.functional.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=False)

        output = self.cls_seg(output)
        return output


class SemanticFPN(nn.Module):
    def __init__(self, configer):
        super(SemanticFPN, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.arch = self.configer.get('network', 'backbone')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.neck = FPN(in_channels=[64, 128, 320, 512], out_channels=256)
        self.decoder = SemanticFPNDecoder(feature_strides=[4, 8, 16, 32],
                                          num_classes=self.num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.decoder(x)

        return x

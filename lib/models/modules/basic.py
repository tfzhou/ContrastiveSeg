import torch
import torch.nn as nn
from collections import OrderedDict


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))

    def forward(self, x):
        return self.block(x)
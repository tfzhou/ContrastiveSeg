##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Jianyuan Guo, Rainbowsecret
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper


class SEModule(nn.Module):
    """Squeeze and Extraction module"""

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module based on DeepLab v3 settings"""

    def __init__(self, in_dim, out_dim, d_rate=[12, 24, 36], bn_type=None):
        super(ASPPModule, self).__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1,
                                          bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[0],
                                          dilation=d_rate[0], bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[1],
                                          dilation=d_rate[1], bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[2],
                                          dilation=d_rate[2], bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )
        self.b4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_dim, out_dim, kernel_size=1,
                                          padding=0, bias=False),
                                ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
                                )

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_dim, out_dim, kernel_size=3, padding=1,
                      bias=False),
            ModuleHelper.BNReLU(out_dim, bn_type=bn_type)
        )

    def forward(self, x):
        h, w = x.size()[2:]
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = F.interpolate(self.b4(x), size=(h, w), mode='bilinear',
                              align_corners=True)

        out = torch.cat((feat0, feat1, feat2, feat3, feat4), dim=1)
        return self.project(out)

class DeepLabHead_MobileNet_V1(nn.Module):
    """Segmentation head based on DeepLab v3"""

    def __init__(self, num_classes, bn_type=None):
        super(DeepLabHead_MobileNet_V1, self).__init__()
        # main pipeline
        self.layer_aspp = ASPPModule(1024, 512, bn_type=bn_type)
        self.refine = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,
                                              padding=1, stride=1, bias=False),
                                    ModuleHelper.BatchNorm2d(bn_type=bn_type)(512),
                                    nn.Conv2d(512, num_classes, kernel_size=1,
                                              stride=1, bias=True))

    def forward(self, x):
        # aspp module
        x_aspp = self.layer_aspp(x)
        # refine module
        x_seg = self.refine(x_aspp)

        return x_seg

class DeepLabHead_MobileNet_V3(nn.Module):
    """Segmentation head based on DeepLab v3"""

    def __init__(self, num_classes, bn_type=None):
        super(DeepLabHead_MobileNet_V3, self).__init__()
        # main pipeline
        self.layer_aspp = ASPPModule(960, 512, bn_type=bn_type)
        self.refine = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,
                                              padding=1, stride=1, bias=False),
                                    ModuleHelper.BatchNorm2d(bn_type=bn_type)(512),
                                    nn.Conv2d(512, num_classes, kernel_size=1,
                                              stride=1, bias=True))

    def forward(self, x):
        # aspp module
        x_aspp = self.layer_aspp(x)
        # refine module
        x_seg = self.refine(x_aspp)

        return x_seg

class DeepLabHead_MobileNet(nn.Module):
    """Segmentation head based on DeepLab v3"""

    def __init__(self, num_classes, bn_type=None):
        super(DeepLabHead_MobileNet, self).__init__()
        # main pipeline
        self.layer_aspp = ASPPModule(1280, 512, bn_type=bn_type)
        self.refine = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,
                                              padding=1, stride=1, bias=False),
                                    ModuleHelper.BatchNorm2d(bn_type=bn_type)(512),
                                    nn.Conv2d(512, num_classes, kernel_size=1,
                                              stride=1, bias=True))

    def forward(self, x):
        # aspp module
        x_aspp = self.layer_aspp(x)
        # refine module
        x_seg = self.refine(x_aspp)

        return x_seg


class DeepLabHead(nn.Module):
    """Segmentation head based on DeepLab v3"""

    def __init__(self, num_classes, bn_type=None):
        super(DeepLabHead, self).__init__()
        # auxiliary loss
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3,
                                                 stride=1, padding=1),
                                       ModuleHelper.BNReLU(256, bn_type=bn_type),
                                       nn.Conv2d(256, num_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True))
        # main pipeline
        self.layer_aspp = ASPPModule(2048, 512, bn_type=bn_type)
        self.refine = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,
                                              padding=1, stride=1, bias=False),
                                    ModuleHelper.BatchNorm2d(bn_type=bn_type)(512),
                                    nn.Conv2d(512, num_classes, kernel_size=1,
                                              stride=1, bias=True))

    def forward(self, x):
        # auxiliary supervision
        x_dsn = self.layer_dsn(x[2])
        # aspp module
        x_aspp = self.layer_aspp(x[3])
        # refine module
        x_seg = self.refine(x_aspp)

        return [x_seg, x_dsn]


class Decoder_Module(nn.Module):

    def __init__(self, bn_type=None, inplane1=512, inplane2=256, outplane=128):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplane1, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplane2, 48, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(48, bn_type=bn_type),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, outplane, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(outplane, bn_type=bn_type),
            nn.Conv2d(outplane, outplane, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(outplane, bn_type=bn_type),
        )

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        return x


class CE2P_Decoder_Module(nn.Module):

    def __init__(self, num_classes, dropout=0, bn_type=None, inplane1=512, inplane2=256):
        super(CE2P_Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplane1, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplane2, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(48, bn_type=bn_type),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            nn.Dropout2d(dropout),
        )

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x

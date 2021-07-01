import torch.nn as nn
from lib.models.tools.module_helper import ModuleHelper


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            ModuleHelper.BNReLU(inter_channels, bn_type='torchsyncbn'),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)

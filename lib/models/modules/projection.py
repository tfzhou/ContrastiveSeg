import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.tools.module_helper import ModuleHelper
from lib.utils.tools.logger import Logger as Log


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)
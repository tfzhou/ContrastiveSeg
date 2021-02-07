import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.projection import ProjectionHead
from lib.models.modules.decoder_block import DeepLabHead
from lib.models.tools.module_helper import ModuleHelper
from lib.utils.tools.logger import Logger as Log
# from lib.extensions.inplace_abn.bn import InPlaceABNSync

class DeepLabV3Contrast(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3Contrast, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.proj_dim = self.configer.get('contrast', 'proj_dim')

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        self.proj_head = ProjectionHead(dim_in=in_channels[1], proj_dim=self.proj_dim)

        self.decoder = DeepLabHead(num_classes=self.num_classes, bn_type=self.configer.get('network', 'bn_type'))

        for modules in [self.proj_head, self.decoder]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                # elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABNSync):
                #     m.weight.data.fill_(1)
                #     m.bias.data.zero_()

    def forward(self, x_, with_embed=False, is_eval=False):
        x = self.backbone(x_)

        embedding = None
        if with_embed and is_eval is False:
            embedding = self.proj_head(x[-1])

        x = self.decoder(x[-4:])

        return {'embedding': embedding, 'seg_aux': x[1], 'seg': x[0]}

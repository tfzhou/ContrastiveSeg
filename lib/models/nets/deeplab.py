import torch.nn as nn

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.decoder_block import DeepLabHead, DeepLabHead_MobileNet, DeepLabHead_MobileNet_V3, DeepLabHead_MobileNet_V1

class DeepLabV3_MobileNetV1(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3_MobileNetV1, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        self.decoder = DeepLabHead_MobileNet_V1(num_classes=self.num_classes,
                                                bn_type=self.configer.get('network', 'bn_type'))

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_):
        x = self.backbone(x_)

        x = self.decoder(x)

        return x

class DeepLabV3_MobileNetV3(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3_MobileNetV3, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        self.decoder = DeepLabHead_MobileNet_V3(num_classes=self.num_classes,
                                                bn_type=self.configer.get('network', 'bn_type'))

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_):
        x = self.backbone(x_)

        x = self.decoder(x)

        return x


class DeepLabV3_MobileNet(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3_MobileNet, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        self.decoder = DeepLabHead_MobileNet(num_classes=self.num_classes,
                                             bn_type=self.configer.get('network', 'bn_type'))

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_):
        x = self.backbone(x_)

        x = self.decoder(x)

        return x


class DeepLabV3(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        self.decoder = DeepLabHead(num_classes=self.num_classes, bn_type=self.configer.get('network', 'bn_type'))

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_):
        x = self.backbone(x_)

        x = self.decoder(x[-4:])

        return x[1], x[0]

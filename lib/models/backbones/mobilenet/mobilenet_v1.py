import torch.nn as nn
from lib.models.tools.module_helper import ModuleHelper
from lib.utils.tools.logger import Logger as Log

class MobileNetV1(nn.Module):
    def __init__(self, alpha=1.0, input_size=224, include_top=False):
        self.alpha = alpha
        self.input_size = input_size
        self.include_top = include_top
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            oup = int(oup * self.alpha)
            return nn.Sequential(
                nn.ConstantPad2d((0, 1, 0, 1), 0),
                nn.Conv2d(inp, oup, 3, stride, 0, bias=False),
                nn.BatchNorm2d(oup, eps=1e-3),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            inp = int(inp * self.alpha)
            oup = int(oup * self.alpha)
            if stride == 2:
                return nn.Sequential(
                    # DepthwiseConv2D
                    nn.ConstantPad2d((0, 1, 0, 1), 0),
                    nn.Conv2d(inp, inp, 3, stride, 0, groups=inp, bias=False),
                    nn.BatchNorm2d(inp, eps=1e-3),
                    nn.ReLU6(inplace=True),

                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup, eps=1e-3),
                    nn.ReLU6(inplace=True),
                )
            else:
                return nn.Sequential(
                    # DepthwiseConv2D
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp, eps=1e-3),
                    nn.ReLU6(inplace=True),

                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup, eps=1e-3),
                    nn.ReLU6(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

    def forward(self, x):
        x = self.model(x)

        return x


class MobileNetV1Backbone(object):
    def __init__(self, configer):
        self.configer = configer

    def __call__(self):
        arch = self.configer.get('network', 'backbone')
        resume = self.configer.get('network', 'resume')

        if arch == 'mobilenet_v1':
            arch_net = MobileNetV1()
            if resume is None:
                arch_net = ModuleHelper.load_model(arch_net,
                                                   pretrained=self.configer.get('network', 'pretrained'),
                                                   all_match=False,
                                                   network='mobilenet_v1')

        else:
            raise Exception('Architecture undefined!')

        return arch_net

import torch
from torch import nn
from lib.models.tools.module_helper import ModuleHelper
from lib.models.backbones.backbone_selector import BackboneSelector
from collections import OrderedDict
import torch.nn.functional as F


class OCR_block(nn.Module):
    """
    Some of the code in this class is borrowed from:
    https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
    """

    def __init__(self, configer, high_level_ch):
        super(OCR_block, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        ocr_mid_channels = 256
        ocr_key_channels = 128
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(high_level_ch, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(ocr_mid_channels, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.spatial_ocr_block import SpatialGather_Module
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Module
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=self.configer.get('network', 'bn_type'))

        self.cls_head = nn.Conv2d(ocr_mid_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(
            nn.Conv2d(high_level_ch, 256, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(256, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, high_level_features):
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats


def make_attn_head(in_ch, out_ch, bn_type=None):
    bot_ch = 256

    od = OrderedDict([('conv0', nn.Conv2d(in_ch, bot_ch, kernel_size=3,
                                          padding=1, bias=False)),
                      ('bn0', ModuleHelper.BatchNorm2d(bn_type=bn_type)(bot_ch)),
                      ('re0', nn.ReLU(inplace=True))])

    if True:  # cfg.MODEL.MSCALE_INNER_3x3:
        od['conv1'] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1,
                                bias=False)
        od['bn1'] = ModuleHelper.BatchNorm2d(bn_type=bn_type)(bot_ch)
        od['re1'] = nn.ReLU(inplace=True)

    if False:  # cfg.MODEL.MSCALE_DROPOUT:
        od['drop'] = nn.Dropout(0.5)

    od['conv2'] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od['sig'] = nn.Sigmoid()

    attn_head = nn.Sequential(od)
    # init_attn(attn_head)
    return attn_head


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)


def fmt_scale(prefix, scale):
    """
    format scale name
    :prefix: a string that is the beginning of the field name
    :scale: a scale value (0.25, 0.5, 1.0, 2.0)
    """

    scale_str = str(float(scale))
    scale_str.replace('.', '')
    return f'{prefix}_{scale_str}x'


class MscaleOCR(nn.Module):
    """
    OCR net
    """

    def __init__(self, configer, criterion=None):
        super(MscaleOCR, self).__init__()
        self.configer = configer
        self.backbone = BackboneSelector(configer).get_backbone()
        self.ocr = OCR_block(configer, 720)
        self.scale_attn = make_attn_head(in_ch=256, out_ch=1, bn_type=self.configer.get('network', 'bn_type'))

    def _fwd(self, x):
        x_size = x.size()[2:]

        x = self.backbone(x)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        high_level_features = torch.cat([feat1, feat2, feat3, feat4], 1)
        cls_out, aux_out, ocr_mid_feats = self.ocr(high_level_features)
        attn = self.scale_attn(ocr_mid_feats)

        aux_out = Upsample(aux_out, x_size)
        cls_out = Upsample(cls_out, x_size)
        attn = Upsample(attn, x_size)

        return {'cls_out': cls_out,
                'aux_out': aux_out,
                'logit_attn': attn}

    def nscale_forward(self, inputs, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.
        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:
              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint
        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.
        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask
        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs['images']

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)

        pred = None
        aux = None
        output_dict = {}

        for s in scales:
            x = torch.nn.functional.interpolate(x_1x, scale_factor=s, mode='bilinear', align_corners=False,
                                                recompute_scale_factor=True)
            outs = self._fwd(x)
            cls_out = outs['cls_out']
            attn_out = outs['logit_attn']
            aux_out = outs['aux_out']

            output_dict[fmt_scale('pred', s)] = cls_out
            if s != 2.0:
                output_dict[fmt_scale('attn', s)] = attn_out

            if pred is None:
                pred = cls_out
                aux = aux_out
            elif s >= 1.0:
                # downscale previous
                pred = torch.nn.functional.interpolate(pred, size=(cls_out.size(2), cls_out.size(3)), mode='bilinear',
                                                       align_corners=False)
                pred = attn_out * cls_out + (1 - attn_out) * pred
                aux = torch.nn.functional.interpolate(aux, size=(cls_out.size(2), cls_out.size(3)), mode='bilinear',
                                                      align_corners=False)
                aux = attn_out * aux_out + (1 - attn_out) * aux
            else:
                # s < 1.0: upscale current
                cls_out = attn_out * cls_out
                aux_out = attn_out * aux_out

                cls_out = torch.nn.functional.interpolate(cls_out, size=(pred.size(2), pred.size(3)), mode='bilinear',
                                                          align_corners=False)
                aux_out = torch.nn.functional.interpolate(aux_out, size=(pred.size(2), pred.size(3)), mode='bilinear',
                                                          align_corners=False)
                attn_out = torch.nn.functional.interpolate(attn_out, size=(pred.size(2), pred.size(3)), mode='bilinear',
                                                           align_corners=False)

                pred = cls_out + (1 - attn_out) * pred
                aux = aux_out + (1 - attn_out) * aux

        output_dict['pred'] = pred
        return output_dict

    def two_scale_forward(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output
        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        x_1x = inputs

        x_lo = torch.nn.functional.interpolate(x_1x, scale_factor=0.5, mode='bilinear',
                                               align_corners=False, recompute_scale_factor=True)
        lo_outs = self._fwd(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        logit_attn = lo_outs['logit_attn']
        attn_05x = logit_attn

        hi_outs = self._fwd(x_1x)
        pred_10x = hi_outs['cls_out']
        p_1x = pred_10x
        aux_1x = hi_outs['aux_out']

        p_lo = logit_attn * p_lo
        aux_lo = logit_attn * aux_lo
        p_lo = torch.nn.functional.interpolate(p_lo, size=(p_1x.size(2), p_1x.size(3)), mode='bilinear',
                                               align_corners=False)
        aux_lo = torch.nn.functional.interpolate(aux_lo, size=(p_1x.size(2), p_1x.size(3)), mode='bilinear',
                                                 align_corners=False)

        logit_attn = torch.nn.functional.interpolate(logit_attn, size=(p_1x.size(2), p_1x.size(3)), mode='bilinear',
                                                     align_corners=False)

        # combine lo and hi predictions with attention
        joint_pred = p_lo + (1 - logit_attn) * p_1x
        joint_aux = aux_lo + (1 - logit_attn) * aux_1x

        output_dict = {
            'pred': joint_pred,
            'aux': joint_aux,
            'pred_05x': pred_05x,
            'pred_10x': pred_10x,
            'attn_05x': attn_05x,
        }
        return output_dict

    def forward(self, inputs):

        # if not self.training:
        #     return self.nscale_forward(inputs, [0.5, 1.0, 2.0])

        return self.two_scale_forward(inputs)

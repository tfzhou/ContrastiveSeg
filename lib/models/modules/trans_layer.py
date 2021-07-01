import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from lib.models.modules.basic import SeparableConv2d


def make_sine_position_embedding(d_model, size, temperature=10000,
                                 scale=2 * math.pi):
    h, w = size, size
    area = torch.ones(1, h, w)  # [b, h, w]
    y_embed = area.cumsum(1, dtype=torch.float32)
    x_embed = area.cumsum(2, dtype=torch.float32)

    one_direction_feats = d_model // 2

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
    pos = pos.flatten(2).permute(0, 2, 1).contiguous()
    return pos


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, low_feature, h_feature, H, W):
        B, N, C = h_feature.shape
        q = self.q(h_feature).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = low_feature.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(low_feature).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        low_feature = (attn @ v).transpose(1, 2).reshape(B, N, C)
        low_feature = self.proj(low_feature)
        low_feature = self.proj_drop(low_feature)

        return low_feature


class SubPixelConv(nn.Module):
    def __init__(self, img_size=224, patch_size=2, in_chans=768, embed_dim=768):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.upsample = nn.Upsample(scale_factor=self.patch_size[0], align_corners=False, mode='bilinear')
        self.upsample_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, bias=True)
        # self.upsample_proj = SeparableConv2d(in_chans, embed_dim, 3)
        # self.upsample_proj = nn.Sequential(
        #     nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1, bias=True),
        #     ModuleHelper.BNReLU(in_chans, bn_type='torchbn'),
        #     nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        # )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(self, x, norm=True):
        B, C, H, W = x.shape

        x = self.upsample(x)
        x = self.upsample_proj(x).flatten(2).transpose(1, 2)
        if norm:
            x = self.norm(x)

        H, W = H * self.patch_size[0], W * self.patch_size[1]

        return x, (H, W)


class ImmediaUpsample(nn.Module):
    def __init__(self, factor=2, in_chans=768, embed_dim=768, num_classes=60):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=num_classes, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear')

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)

        return x


class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, num_heads, mlp_ratio, sr_ratio):
        super(AlignedModule, self).__init__()
        self.reset_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h, w = low_feature.size()[2:]
        size = (h, w)

        h_feature = self.reset_h(h_feature)
        h_feature_orign = h_feature
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow_in = torch.cat([h_feature, low_feature], 1)
        flow = self.flow_make(flow_in)
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature.flatten(2).transpose(1, 2), (h, w)

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

# att low_feature + flow wrap high feature
# class AlignedModule(nn.Module):

#     def __init__(self, inplane, outplane, num_heads, mlp_ratio, sr_ratio):
#         super(AlignedModule, self).__init__()
#         self.reset_h = nn.Conv2d(inplane, outplane, 1, bias=False)
#         self.norm_h = partial(nn.LayerNorm, eps=1e-6)(outplane)
#         self.norm_l = partial(nn.LayerNorm, eps=1e-6)(outplane)
#         self.context_att = Attention(dim=outplane, num_heads=num_heads, sr_ratio=sr_ratio)
#         self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1,  bias=False)

#     def forward(self, x):
#         low_feature, h_feature = x
#         B, _, h, w = low_feature.size()
#         size = (h, w)

#         h_feature = self.reset_h(h_feature)
#         h_feature_orign = h_feature
#         h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
#         low_feature = self.context_att(self.norm_l(low_feature.flatten(2).transpose(1, 2)), self.norm_h(h_feature.flatten(2).transpose(1, 2)), h, w)
#         low_feature = low_feature.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()

#         flow_in = torch.cat([h_feature, low_feature], 1)
#         flow = self.flow_make(flow_in)
#         h_feature = self.flow_warp(h_feature_orign, flow, size=size)

#         return low_feature, h_feature.flatten(2).transpose(1, 2), (h, w)

#     def flow_warp(self, input, flow, size):
#         out_h, out_w = size
#         n, c, h, w = input.size()

#         norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
#         h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
#         w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
#         grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
#         grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
#         grid = grid + flow.permute(0, 2, 3, 1) / norm

#         output = F.grid_sample(input, grid)
#         return output

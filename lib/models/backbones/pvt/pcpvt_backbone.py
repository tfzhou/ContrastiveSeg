from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.tools.module_helper import ModuleHelper

__all__ = [
    'pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large'
]


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        self.blocks = nn.ModuleList()

        # patch_embed
        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(
                    PatchEmbed(img_size // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        # pos_embed
        # self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        # self.pos_drop1 = nn.Dropout(p=drop_rate)
        # self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        # self.pos_drop2 = nn.Dropout(p=drop_rate)
        # self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        # self.pos_drop3 = nn.Dropout(p=drop_rate)
        # self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, embed_dims[3]))
        # self.pos_drop4 = nn.Dropout(p=drop_rate)
        self.pos_block = nn.ModuleList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims]
        )

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([Block(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k])
                for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        # init weights
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

    def no_weight_decay(self):
        return set(['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def forward_features(self, x):
        outs = []
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            outs.append(x)

        return outs

    def forward(self, x):
        outs = self.forward_features(x)

        return outs


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def pvt_tiny(configer, **kwargs):
    img_size = configer.get('train', 'data_transformer')['input_size'][0]
    num_classes = configer.get('data', 'num_classes')
    model = PyramidVisionTransformer(img_size=img_size, num_classes=num_classes,
                                     patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
                                     sr_ratios=[8, 4, 2, 1], drop_rate=0.1,
                                     drop_path_rate=0.1,
                                     **kwargs)
    return model


def pvt_small(configer, **kwargs):
    img_size = configer.get('train', 'data_transformer')['input_size'][0]
    num_classes = configer.get('data', 'num_classes')
    model = PyramidVisionTransformer(img_size=img_size, num_classes=num_classes,
                                     patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
                                     sr_ratios=[8, 4, 2, 1], drop_rate=0.0,
                                     drop_path_rate=0.1,
                                     **kwargs)

    return model


def pvt_medium(configer, **kwargs):
    img_size = configer.get('train', 'data_transformer')['input_size'][0]
    num_classes = configer.get('data', 'num_classes')
    model = PyramidVisionTransformer(img_size=img_size, num_classes=num_classes,
                                     patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
                                     sr_ratios=[8, 4, 2, 1],
                                     # drop_rate=0.0, drop_path_rate=0.05)
                                     **kwargs)

    return model


def pvt_large(configer, **kwargs):
    img_size = configer.get('train', 'data_transformer')['input_size'][0]
    num_classes = configer.get('data', 'num_classes')
    model = PyramidVisionTransformer(img_size=img_size, num_classes=num_classes,
                                     patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                     mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3],
                                     sr_ratios=[8, 4, 2, 1],
                                     # drop_rate=0.0, drop_path_rate=0.02)
                                     **kwargs)

    return model


class PCPVTBackbone(object):
    def __init__(self, configer):
        self.configer = configer

    def __call__(self):
        arch = self.configer.get('network', 'backbone')

        if arch == 'pcpvt_tiny':
            model = pvt_tiny(configer=self.configer)
        elif arch == 'pcpvt_small':
            model = pvt_small(configer=self.configer)
        elif arch == 'pcpvt_medium':
            model = pvt_medium(configer=self.configer)
        elif arch == 'pcpvt_large':
            model = pvt_large(configer=self.configer)

        model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'),
                                        all_match=False, network="pcpvt")

        return model

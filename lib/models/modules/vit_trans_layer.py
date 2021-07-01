import torch
import torch.nn as nn
from lib.models.modules.norm import trunc_normal_
import math
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    # two mlp, fc-relu-drop-fc-relu-drop
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


class Attention_Encoder(nn.Module):
    def __init__(self, dim, kv_reduced_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if kv_reduced_dim is not None and type(kv_reduced_dim) == int:
            self.fc_k = nn.Linear()

    def forward(self, x):
        B, N, C = x.shape
        # qkv shape [3, N, num_head, HW, C//num_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [N, num_head, HW, C//num_head]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_Decoder(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.fc_q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.fc_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, x):
        # q:[B,12,256] x:[B,HW,256]
        B, N, C = x.shape
        n_class = q.shape[1]

        q = self.fc_q(q).reshape(B, self.num_heads, n_class, C // self.num_heads)
        kv = self.fc_kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, num_head, HW, 256/num_head]

        attn1 = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_head, 12, HW]
        attn2 = attn1.softmax(dim=-1)
        attn3 = self.attn_drop(attn2)  # [B, num_head, 11, HW]

        x = (attn3 @ v).reshape(B, n_class, C)
        x = self.proj(x)
        x = self.proj_drop(x)  # [B, 12, 256]

        # attn = attn3.permute(0, 2, 1, 3)
        attn = attn1.permute(0, 2, 1, 3)
        # attn = attn2.permute(0, 2, 1, 3)
        return attn, x


class Block_Encoder(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_Encoder(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_Decoder(nn.Module):

    def __init__(self, dim, num_heads, feat_HxW, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_clsembed = norm_layer(dim)

        self.attn = Attention_Decoder(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(256)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp3 = Mlp(in_features=feat_HxW, hidden_features=feat_HxW * 3, act_layer=act_layer, drop=drop)

    def forward(self, query, feat):
        # query:[B,12,256] feat:[B,12,HW]
        attn, query = self.attn(self.norm1_clsembed(query), self.norm1(feat))
        query = query + self.drop_path(query)
        query = query + self.drop_path(self.mlp(self.norm2(query)))

        feat = feat + self.drop_path(feat)
        feat = feat + self.drop_path(self.mlp2(self.norm3(feat)))

        attn = attn + self.drop_path(attn)
        attn = attn + self.drop_path(self.mlp3(self.norm4(attn)))

        return attn, query, feat


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, input_dim=2048, embed_dim=768, depth=12, num_patches=32 * 32, nclass=12,
                 decoder_feat_HxW=1024, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.cls_embed = nn.Parameter(torch.zeros(1, nclass, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks_encoder = nn.ModuleList([
            Block_Encoder(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks_decoder = nn.ModuleList([
            Block_Decoder(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, feat_HxW=decoder_feat_HxW, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.hybrid_embed = HybridEmbed(input_dim, embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'cls_embed'}

    def forward_encoder(self, x, h, w):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        pos_embed = self.resize_pos_embed(x, pos_embed, h, w)
        x = x + pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks_encoder:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def resize_pos_embed(self, x, pos_embed, h, w):
        # if x.shape[1] == pos_embed.shape[1]:
        #     return pos_embed

        # n, hw, c = x.shape
        # x_h = x_w = int(math.sqrt(hw - 1))
        # assert x_h * x_w == hw - 1

        cls_pos_embed, feat_pos_embed = pos_embed[:, 0:1, :], pos_embed[:, 1:, :]
        feat_h = feat_w = int(math.sqrt(feat_pos_embed.shape[1]))
        assert feat_h * feat_w == feat_pos_embed.shape[1]
        feat_pos_embed = feat_pos_embed.reshape(feat_pos_embed.shape[0], feat_h, feat_w, -1).permute(0, 3, 1,
                                                                                                     2)  # [n,c,h,w]
        feat_pos_embed = F.interpolate(feat_pos_embed, (h, w), mode='bilinear', align_corners=True).permute(0, 2, 3,
                                                                                                            1) \
            .reshape(feat_pos_embed.shape[0], h * w, -1)

        new_pos_embed = torch.cat([cls_pos_embed, feat_pos_embed], dim=1)
        assert new_pos_embed.shape[1] == x.shape[1]
        return new_pos_embed

    def forward_decoder(self, x):
        attns_list = []
        feat = x
        B = feat.shape[0]

        for idx, blk in enumerate(self.blocks_decoder):
            if idx == 0:
                query = self.cls_embed.expand(B, -1, -1)
            else:
                query += self.cls_embed.expand(B, -1, -1)
            attn, query, feat = blk(query, feat)
            attns_list.append(attn)

        return attns_list

    def forward(self, x, use_decoder=False):
        '''
        x: [N,C,H,W]
        '''
        pass

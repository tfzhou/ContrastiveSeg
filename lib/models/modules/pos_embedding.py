import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def initialize_embedding(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                module.weight.data.zero_()  # original


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        if d_hid > 50:
            cycle = 10
        elif d_hid > 5:
            cycle = 100
        else:
            cycle = 10000
        cycle = 10 if d_hid > 50 else 100
        return position / np.power(cycle, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)


class PosEmbedding2D(nn.Module):

    def __init__(self, pos_rfactor, dim):
        super(PosEmbedding2D, self).__init__()

        self.pos_layer_h = nn.Embedding((128 // pos_rfactor) + 1, dim)
        self.pos_layer_w = nn.Embedding((128 // pos_rfactor) + 1, dim)
        initialize_embedding(self.pos_layer_h)
        initialize_embedding(self.pos_layer_w)

    def forward(self, x, pos):
        pos_h, pos_w = pos
        pos_h = pos_h.unsqueeze(1)
        pos_w = pos_w.unsqueeze(1)
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2:], mode='nearest').long()  # B X 1 X H X W
        pos_w = nn.functional.interpolate(pos_w.float(), size=x.shape[2:], mode='nearest').long()  # B X 1 X H X W
        pos_h = self.pos_layer_h(pos_h).transpose(1, 4).squeeze(4)  # B X 1 X H X W X C
        pos_w = self.pos_layer_w(pos_w).transpose(1, 4).squeeze(4)  # B X 1 X H X W X C
        x = x + pos_h + pos_w
        return x


class PosEncoding1D(nn.Module):

    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEncoding1D, self).__init__()
        print("use PosEncoding1D")
        self.sel_index = torch.tensor([0]).cuda()
        pos_enc = (get_sinusoid_encoding_table((128 // pos_rfactor) + 1, dim) + 1)
        self.pos_layer = nn.Embedding.from_pretrained(embeddings=pos_enc, freeze=True)
        self.pos_noise = pos_noise
        self.noise_clamp = 16 // pos_rfactor  # 4: 4, 8: 2, 16: 1

        self.pos_rfactor = pos_rfactor
        if pos_noise > 0.0:
            self.min = 0.0  # torch.tensor([0]).cuda()
            self.max = 128 // pos_rfactor  # torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        pos_h, _ = pos  # B X H X W
        pos_h = pos_h // self.pos_rfactor
        pos_h = pos_h.index_select(2, self.sel_index).unsqueeze(1).squeeze(3)  # B X 1 X H
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long()  # B X 1 X 48

        if self.training is True and self.pos_noise > 0.0:
            # pos_h = pos_h + (self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long()
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3).cuda() // 1).long(),
                                        min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)
            # pos_h = torch.where(pos_h < self.min_tensor, self.min_tensor, pos_h)
            # pos_h = torch.where(pos_h > self.max_tensor, self.max_tensor, pos_h)

        pos_h = self.pos_layer(pos_h).transpose(1, 3).squeeze(3)  # B X 1 X 48 X 80 > B X 80 X 48 X 1
        x = x + pos_h
        if return_posmap:
            return x, self.pos_layer.weight  # 33 X 80
        return x


class PosEmbedding1D(nn.Module):

    def __init__(self, pos_rfactor, dim, pos_noise=0.0):
        super(PosEmbedding1D, self).__init__()
        print("use PosEmbedding1D")
        self.sel_index = torch.tensor([0]).cuda()
        self.pos_layer = nn.Embedding((128 // pos_rfactor) + 1, dim)
        initialize_embedding(self.pos_layer)
        self.pos_noise = pos_noise
        self.pos_rfactor = pos_rfactor
        self.noise_clamp = 16 // pos_rfactor  # 4: 4, 8: 2, 16: 1

        if pos_noise > 0.0:
            self.min = 0.0  # torch.tensor([0]).cuda()
            self.max = 128 // pos_rfactor  # torch.tensor([128//pos_rfactor]).cuda()
            self.noise = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([pos_noise]))

    def forward(self, x, pos, return_posmap=False):
        pos_h, _ = pos  # B X H X W
        pos_h = pos_h // self.pos_rfactor
        pos_h = pos_h.index_select(2, self.sel_index).unsqueeze(1).squeeze(3)  # B X 1 X H
        pos_h = nn.functional.interpolate(pos_h.float(), size=x.shape[2], mode='nearest').long()  # B X 1 X 48

        if self.training is True and self.pos_noise > 0.0:
            # pos_h = pos_h + (self.noise.sample(pos_h.shape).squeeze(3).cuda()//1).long()
            pos_h = pos_h + torch.clamp((self.noise.sample(pos_h.shape).squeeze(3).cuda() // 1).long(),
                                        min=-self.noise_clamp, max=self.noise_clamp)
            pos_h = torch.clamp(pos_h, min=self.min, max=self.max)

        pos_h = self.pos_layer(pos_h).transpose(1, 3).squeeze(3)  # B X 1 X 48 X 80 > B X 80 X 48 X 1
        x = x + pos_h
        if return_posmap:
            return x, self.pos_layer.weight  # 33 X 80
        return x

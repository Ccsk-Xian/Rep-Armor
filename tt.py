# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# # kernel_size=3
# # sigma = 0.5
# # # 定义1D高斯函数
# # def gaussian_1d(d, sigma):
# #     # return (1 / (math.sqrt(2 * math.pi) * sigma)) * torch.exp(-0.5 * (d / sigma) ** 2)
# #     return torch.exp(-0.5 * (d**2) / sigma**2)


# # dist_x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
# # print(dist_x)
# # dist_y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
# # mask_1d = gaussian_1d(dist_x, sigma)
# # grid_x, grid_y = torch.meshgrid(dist_x, dist_y,indexing='xy')
# # print(grid_x)
    
# #     # 生成2D高斯掩膜（通过外积计算）
# # mask_2d = mask_1d.view(-1, 1) * mask_1d.view(1, -1)
# # gaussian = torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma**2)
# # print(mask_2d)
# # print(gaussian)
# # g = torch.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
# # print(g)
# bsz=2
# f_s = torch.tensor([2.,5.,1.,3.])
# f_t = torch.tensor([3.,4.,1.,2.])
# f_s = f_s.view(bsz, -1)
# f_t = f_t.view(bsz, -1)
# print(f_s)
# # c = F.softmax(a)
# # d = F.log_softmax(a)
# # print(c)
# # print(d)
# G_s = torch.mm(f_s, torch.t(f_s))
# print(G_s)
# # G_s = G_s / G_s.norm(2)
# G_s = torch.nn.functional.normalize(G_s)
# print(G_s)
# G_t = torch.mm(f_t, torch.t(f_t))
# # G_t = G_t / G_t.norm(2)
# G_t = torch.nn.functional.normalize(G_t)

# G_diff = G_t - G_s
# loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
# print(loss)

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    print(gamma.shape)
    print(kernel.shape)
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps
    
in_channels = 3
kernel_size = 3
internal_channels_1x1_3x3 = 16
padding = kernel_size //2
conv1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                            kernel_size=kernel_size, stride=1, padding=0, groups=1, bias=False)

# bn1  = BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True)
bn1 = nn.BatchNorm2d(internal_channels_1x1_3x3)

# k_origin, b_origin = transI_fusebn(conv1.weight, bn1)
# print(kernel_size,b_origin)

c = nn.Sequential(conv1,bn1)
x = torch.ones(1, 3, 224, 224)
output = c(x)
print(output)
print(output.shape)
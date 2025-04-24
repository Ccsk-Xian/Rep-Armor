import time
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


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

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        groups=groups, bias=False))
    result.add_module('conv_1', nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                        kernel_size=3, stride=1, padding=1,
                                        groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std

def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)

def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)

def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        print(';')
        print(k2.shape)
        print(k1.permute(1, 0, 2, 3).shape)
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      #
        print(k.shape)
        print(b1.reshape(1, -1, 1, 1).shape)
        dd = k2 * b1.reshape(1, -1, 1, 1)
        print('1123')
        print(dd.shape)
        b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.permute(1, 0, 2, 3)
        k1_group_width = k1.size(0) // groups
        k2_group_width = k2.size(0) // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g*k1_group_width:(g+1)*k1_group_width, :, :]
            k2_slice = k2[g*k2_group_width:(g+1)*k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g*k1_group_width:(g+1)*k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transIII_kxk_1x1(k1, b1, k2, b2, groups):
    if groups == 1:
        # print(';')
        # print(k2.shape) 64 128 1 1
        # print(k1.shape)  128 64 3 3
        k = F.conv2d(k1.permute(1, 0, 2, 3), k2).transpose(0,1)      #
        print(k1.shape)
        print(b2.reshape(1, -1, 1, 1).shape)
        dd = k1 * b2.reshape(1, -1, 1, 1)
        print('4332')
        print(dd.shape)
        b_hat = (k1 * b2.reshape(1, -1, 1, 1)).sum((0, 2, 3))
        print(b_hat.shape)
    else:
        k_slices = []
        b_slices = []
        k2_T = k2.permute(1, 0, 2, 3)
        k2_group_width = k2.size(0) // groups
        k1_group_width = k1.size(0) // groups
        for g in range(groups):
            k2_T_slice = k2_T[:, g*k2_group_width:(g+1)*k2_group_width, :, :]
            k1_slice = k1[g*k1_group_width:(g+1)*k1_group_width, :, :, :]
            k_slices.append(F.conv2d(k1_slice, k2_T_slice))
            b_slices.append((k1_slice * b1[g*k2_group_width:(g+1)*k2_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2



def main():
    f1 = torch.randn(1, 64, 64, 64)
    # block = RepNeedleBlock(in_channels=64, out_channels=64)
    in_channels=64
    out_channels=64
    a1 =  nn.Conv2d(in_channels,128,1,1,0,bias=False,groups=1)
    b1 =   nn.BatchNorm2d(128)
    # b1 = BNAndPadLayer(pad_pixels=0, num_features=128)
    a = nn.Conv2d(128,128,3,1,1,bias=False,groups=1)
    b =   nn.BatchNorm2d(128)
    f = nn.Conv2d(128,64,1,1,0,bias=False,groups=1)
    e =   nn.BatchNorm2d(64)
    block = nn.Sequential(a1,b1,a,b,f,e)
    # block = TestModel(in_channels=64, out_channels=64,groups=1)
    # print(block)
    # print(len(block.needle))
    # print(block.needle[0].weight)
    block.eval()
    with torch.no_grad():
        output1 = block(f1)
        start_time = time.time()
        for _ in range(1):
            block(f1)
        print(f"consume time: {time.time() - start_time}")
        total_params1 = sum(p.numel() for p in block.parameters())
        print('total_params_before_merge:'+str(total_params1))
        # re-parameterization
        # block.switch_to_deploy()
        # block.needle.switch_to_deploy()
        k1,bias1=transI_fusebn(a1.weight,b1)
        # print(a.weight.shape)
        k,bias=transI_fusebn(a.weight,b)
        k2,bias2 =transI_fusebn(f.weight,e)
        # print(k.shape)
        k,bias = transIII_1x1_kxk(k1,bias1,k,bias,1)
        print(bias.shape)
        k,bias = transIII_kxk_1x1(k,bias,k2,bias2,1)
        print(bias.shape)
        # print(k.shape)
        # print(bias.shape)
        d = nn.Conv2d(in_channels,out_channels,3,1,1,bias=True,groups=1)
        d.weight.data = k
        d.bias.data = bias
        # block.switch_to_deploy1()
        output2 = d(f1)
        start_time = time.time()
        for _ in range(1):
            d(f1)
        print(f"consume time: {time.time() - start_time}")
        total_params2 = sum(p.numel() for p in d.parameters())
        print('total_params_after_merge:'+str(total_params2))
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-03, atol=1e-05)
        print("convert module has been tested, and the inference results, before and after merging, are equal!")
if __name__ == '__main__':
    main()

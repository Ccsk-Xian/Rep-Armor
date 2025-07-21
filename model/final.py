"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>
"""
import torch.nn as nn
import math

import torch
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
import collections.abc as container_abcs
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .convnet_utils import conv_bn_relu

__all__ = ['mobilenetv2_T_w', 'mobile_half']

BN = None
class RepTowerBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,expand_rate = 3,neck_rate=0.5,pattern=3):
        super(RepTowerBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.pattern =pattern
     
        # 不能引入bias，所以只能是identity
        self.skip = nn.Identity()
            
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
        else:
            if pattern ==3:
                self.needle = nn.Sequential(
                    nn.Conv2d(in_channels,int(in_channels*expand_rate),1,1,0,bias=False,groups=groups),
                    # nn.BatchNorm2d(int(in_channels*expand_rate)),
                    nn.Conv2d(int(in_channels*expand_rate),int(in_channels*expand_rate),1,1,0,bias=False,groups=groups),
                    nn.BatchNorm2d(int(in_channels*expand_rate)),
                    nn.Conv2d(int(in_channels*expand_rate),out_channels,1,1,0,bias=False,groups=groups),
                    # nn.BatchNorm2d(out_channels)
                )
            elif pattern ==2:
                self.needle = nn.Sequential(
                    nn.Conv2d(in_channels,int(in_channels*expand_rate),1,1,0,bias=False,groups=groups),
                    # nn.Conv2d(int(in_channels*expand_rate),int(in_channels*expand_rate),1,1,0,bias=False,groups=groups),
                    nn.BatchNorm2d(int(in_channels*expand_rate)),
                    nn.Conv2d(int(in_channels*expand_rate),out_channels,1,1,0,bias=False,groups=groups),
                    # nn.BatchNorm2d(out_channels),
                )
            else:
                self.needle = nn.Sequential(
                    nn.Conv2d(in_channels,out_channels,1,1,0,bias=False,groups=groups),
                    nn.BatchNorm2d(out_channels),
                )


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.rbr_reparam(inputs)
        # if self.groups==1:
        #     return (self.needle(inputs)+self.down_needle(inputs))/2+self.skip(inputs)
        # else:
        return self.needle(inputs)+self.skip(inputs)
        # return inputs
        # return self.needle(inputs)

    def merge_reptower(self):
        if not self.deploy:
            self.deploy=True
            if self.pattern==3:
                middle_k,middle_b = fuse_bn(self.needle[1],self.needle[2])
                k,b = transIII_1x1_kxk(self.needle[0].weight,self.needle[0].bias,middle_k,middle_b,self.needle[1].groups)
                k,b = transIII_1x1_kxk(k,b,self.needle[3].weight,self.needle[3].bias,self.needle[1].groups)
            elif self.pattern==2:
                middle_k,middle_b = fuse_bn(self.needle[0],self.needle[1])
                k,b = transIII_1x1_kxk(middle_k,middle_b,self.needle[2].weight,self.needle[2].bias,self.needle[0].groups)
            else:
                k,b = fuse_bn(self.needle[0],self.needle[1])
            

            # if not hasattr(self, 'id_tensor'):
            input_dim = self.needle[0].in_channels // self.needle[0].groups
            kernel_value = np.zeros((self.needle[0].in_channels, input_dim, 1, 1), dtype=np.float32)
            for i in range(self.needle[0].in_channels):
                kernel_value[i, i % input_dim, 0, 0] = 1
            self.id_tensor = torch.from_numpy(kernel_value).to(self.needle[0].weight.device)
            kernel = self.id_tensor

            k = k+kernel
            if self.pattern==1:
                self.rbr_reparam = nn.Conv2d(in_channels=self.needle[0].in_channels, out_channels=self.needle[0].out_channels,
                                     kernel_size=1, stride=self.needle[0].stride,
                                     padding=0, dilation=1, groups=self.needle[0].groups, bias=True)
            else:

                self.rbr_reparam = nn.Conv2d(in_channels=self.needle[0].in_channels, out_channels=self.needle[-1].out_channels,
                                     kernel_size=1, stride=self.needle[0].stride,
                                     padding=0, dilation=1, groups=self.needle[0].groups, bias=True)
            
            self.rbr_reparam.weight.data = k
            self.rbr_reparam.bias.data = b
            self.__delattr__('needle')
            self.deploy = True
            
class DOConv2d(Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DOConv2d, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.d_diag = Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.d_diag = Parameter(d_diag, requires_grad=False)
        ##################################################################################################

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N)
            D = self.D + self.d_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum('cms,ois->oim', D, W), DoW_shape)
            #######################################################
        else:
            # in this case D_mul == M * N
            # reshape from
            # (out_channels, in_channels // groups, D_mul)
            # to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(self.W, DoW_shape)
        return self._conv_forward(input, DoW)


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
_pair = _ntuple(2)

__all__ = ['mobilenetv2_T_w', 'mobile_half']

BN = None

def conv1_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
from timm.models.vision_transformer import trunc_normal_
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        # self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('drop',torch.nn.Dropout(p=0.2))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    

class Classfier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) 

        self.classifier_dist = BN_Linear(dim, num_classes) 

    def forward(self, x):
        if hasattr(self, 'classifier_dist'):
            return (self.classifier(x)+self.classifier_dist(x))/2
        else:
            return self.classifier(x)


    @torch.no_grad()
    def fuse(self):

        if hasattr(self, 'classifier_dist'):
            self.classifier.l.weight += self.classifier_dist.l.weight
            self.classifier.l.bias += self.classifier_dist.l.bias
            self.classifier.l.weight /= 2.
            self.classifier.l.bias /= 2.
        self.__delattr__('classifier_dist')


def fuse_bn(conv, bn):
        conv_bias = 0 if conv.bias is None else conv.bias
        std = (bn.running_var + bn.eps).sqrt()
        return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def transIV_depthconcat(kernels, biases):
    return torch.cat(kernels, dim=0), torch.cat(biases)


def transIII_kxk_1x1(k1, b1, k2, b2,groups):

    w2_mat = k2.view(k2.size(0), k2.size(1))
    Z = torch.einsum('om,mcij->ocij', w2_mat, k1)
    M, O = k1.size(0), k2.size(0)
    if b1 is None:
        b1_t = torch.zeros(M, device=k1.device)
    else:
        b1_t = b1
    if b2 is None:
        b2_t = torch.zeros(O, device=k2.device)
    else:
        b2_t = b2

    #    b_merged[o] = b2[o] + sum_m W2[o,m,0,0] * b1[m]
    b_merged = b2_t + (w2_mat * b1_t.view(1, -1)).sum(dim=1)
    
    return Z, b_merged



def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        # print(';')
        # print(k2.shape)
        # print(k1.permute(1, 0, 2, 3).shape)
        k = F.conv2d(k2, k1.permute(1, 0, 2, 3))      #
        # print(k.shape)
        # print(b1.reshape(1, -1, 1, 1).shape)
        # dd = k2 * b1.reshape(1, -1, 1, 1)
        # print('1123')
        # print(dd.shape)
        if b1 is not None:
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
    if b1 is not None:
        if b2 is not None:
            return k, b_hat + b2
        else:
            return k,b_hat
    else: 
        return k,b2


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel

class Rep_bn(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, inchannels,outchannels, kernel_size, stride, bias, group):
        super().__init__()
        self.lk_origin = nn.Conv2d(inchannels, outchannels, kernel_size, stride=stride,
                                    padding=kernel_size//2, dilation=1, groups=group, bias=bias,
                                    )
        self.origin_1x1 = RepTowerBlock(outchannels,outchannels,groups=group,expand_rate=4)
   

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]  # 5 17 9 11 15
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3] # 5 13 7 11 15
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]  # 5 13 7 9 11
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]  # 5 9 7 9 11
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]  # 5 9 7 9
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3,3]  # 5 5 7
            self.dilates = [1, 2, 3,1]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3] # 3 5
            self.dilates = [1, 2]
        elif kernel_size == 3:
            self.kernel_sizes = [3] # 3 
            self.dilates = [1]


        if not bias:
            self.origin_bn = nn.BatchNorm2d(outchannels) 
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=k, stride=stride,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=group,
                                           bias=bias)
                                           )
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))
                self.__setattr__('conv_one{}_{}'.format(k, r),
                                RepTowerBlock(inchannels,inchannels,groups=group,expand_rate=4))
                                

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)/math.sqrt(len(self.kernel_sizes))
        out = self.origin_1x1(self.origin_bn(self.lk_origin(x)))

        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            inner_conv1 = self.__getattr__('conv_one{}_{}'.format(k, r))

            out = out +bn(conv(inner_conv1(x)))

        return out/math.sqrt(len(self.kernel_sizes))


    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            self.origin_1x1.merge_reptower()
            origin_k, origin_b = transIII_kxk_1x1(origin_k,origin_b,self.origin_1x1.rbr_reparam.weight,self.origin_1x1.rbr_reparam.bias,self.lk_origin.groups)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                inner_conv1 = self.__getattr__('conv_one{}_{}'.format(k, r))
                inner_conv1.merge_reptower()
                branch_k, branch_b = fuse_bn(conv, bn)
                branch_k, branch_b = transIII_1x1_kxk(inner_conv1.rbr_reparam.weight,inner_conv1.rbr_reparam.bias,branch_k,branch_b,conv.groups)

                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = nn.Conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=1, bias=True,
                                    )
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            self.__delattr__('origin_1x1')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))
                self.__delattr__('conv_one{}_{}'.format(k, r))



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
def conv_bn1(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

def conv_bn_rep(in_channels, out_channels, kernel_size, stride, padding, groups=1,use_bn=True,follow=False,pattern=3):
    result = nn.Sequential()
    if stride == 2 or in_channels!=out_channels:
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
        # result.add_module('conv',DOConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,padding=padding,bias=False),)
    else:
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
        # result.add_module('conv',DOConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,padding=padding,bias=False),)
    if use_bn:
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    if follow:
        result.add_module('conv1',RepTowerBlock(in_channels=out_channels,out_channels=out_channels,groups=groups,pattern=pattern,expand_rate=2))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,use_act=True,use_bn=True,use_identity=True,count=True):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.use_act = use_act
        # self.count = 5 if count else 1

       
        if self.use_act:
            self.nonlinearity = nn.ReLU()
        else:
            self.nonlinearity = nn.Identity()

        if use_se:
            #   We didn't use SEblock here, cause we wanna verify the effectiveness of Rep-Armor, the strenth of specific structural re-parameterization optimization.
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:

            self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=3)
            self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=2)
            self.rbr_dense_3 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=1)
            self.rbr_dense_4 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False)
            self.rbr_dense_5 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False)
            



    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)/5))


        return self.nonlinearity(self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs)+self.rbr_dense_3(inputs)+self.rbr_dense_4(inputs)+self.rbr_dense_5(inputs))/5))

    def merge_recovery(self):
        if not self.deploy:

            origin_k, origin_b = fuse_bn(self.rbr_dense[0], self.rbr_dense[1])
            self.rbr_dense[-1].merge_reptower()
            # origin_k, origin_b = transIII_1x1_kxk(origin_k,origin_b,self.rbr_dense[-1].rbr_reparam.weight,self.rbr_dense[-1].rbr_reparam.bias,self.rbr_dense[0].groups)
            origin_k, origin_b = transIII_kxk_1x1(origin_k,origin_b,self.rbr_dense[-1].rbr_reparam.weight,self.rbr_dense[-1].rbr_reparam.bias,self.rbr_dense[0].groups)
            for i in range (2,6):
                branch = self.__getattr__('rbr_dense_{}'.format(i))
                # print(type(branch[-1]))
                # if branch[-1] is nn.Sequential:
                branch_k, branch_b = fuse_bn(branch[0], branch[1])
                if isinstance(branch[-1], RepTowerBlock):
                    # print('11')
                    branch[-1].merge_reptower()
                    # branch_k, branch_b = transIII_1x1_kxk(branch_k,branch_b,branch[-1].rbr_reparam.weight,branch[-1].rbr_reparam.bias,branch[0].groups)
                    branch_k, branch_b = transIII_kxk_1x1(branch_k,branch_b,branch[-1].rbr_reparam.weight,branch[-1].rbr_reparam.bias,branch[0].groups)

                origin_k += branch_k
                origin_b += branch_b
            # print(self.rbr_dense[0].kernel_size[0])
            self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense[0].in_channels, out_channels=self.rbr_dense[0].out_channels,
                                     kernel_size=self.rbr_dense[0].kernel_size, stride=self.rbr_dense[0].stride,
                                     padding=self.rbr_dense[0].kernel_size[0]//2, dilation=1, groups=self.rbr_dense[0].groups, bias=True)
            
            self.rbr_reparam.weight.data = origin_k
            self.rbr_reparam.bias.data = origin_b
            self.__delattr__('rbr_dense')
            for i in range (2,6):
                self.__delattr__('rbr_dense_{}'.format(i))

            self.deploy = True

    #   Optional. This may improve the accuracy and facilitates quantization in some cases.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        
        
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True






def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,count):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        self.pw = RepVGGBlock(in_channels=inp, out_channels=inp * expand_ratio, groups=1, kernel_size=1, stride=1, padding=0,count=count)
        
        self.dw = conv_bn_relu(inp*expand_ratio,inp*expand_ratio,3,stride,1,1,inp*expand_ratio)
    
        self.conv = nn.Sequential(    
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x
        if self.use_res_connect:
            return t + self.conv(self.dw(self.pw(x)))
        else:
            return self.conv(self.dw(self.pw(x)))
    
    def merge_inverte(self):
        self.pw.merge_recovery()
        self.dw.switch_to_deploy()


class MobileNetV2(nn.Module):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 ):
        super(MobileNetV2, self).__init__()


        # setting of inverted residual blocks  cifar 第二行s为1
        self.interverted_residual_setting = [
            # t, c, n, s,count
            [1, 16, 1, 1,True],
            [T, 24, 2, 2,True],
            [T, 32, 3, 2,True],
            [T, 64, 4, 2,True],
            [T, 96, 3, 1,True],
            [T, 160, 3, 2,True],
            [T, 320, 1, 1,True],
        ]



        # building first layer

        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
       
        # 1. large conv layer is implemented by the Uni block (dilated structural re-parameterization + RepTower + Ensemble Modification)
        self.conv1 = Rep_bn(inchannels=3,outchannels=3,kernel_size=7,stride=1,bias=False,group=1)
        # self.conv1 = Rep_bn(inchannels=3,outchannels=3,kernel_size=3,stride=1,bias=False,group=1)
        self.act = nn.ReLU(inplace=True)
        # 2.the original Stem layer is enhanced by the DBB block
        self.conv1_follow = conv_bn_relu(3,input_channel,3,2,1)
        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s,count in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t,count)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else int(c*width_mult)*4

        self.classifier = Classfier(output_channel, feature_dim)
        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, feature_dim),
        # )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # for compatibility issues of pytorch-onnx-tf-tflite
        self.avgpool = nn.AvgPool2d(kernel_size=(7,7), stride=1)

        self._initialize_weights()
        print(T, width_mult)

    def get_bn_before_relu(self):
        bn1 = self.blocks[1][-1].conv[-1]
        bn2 = self.blocks[2][-1].conv[-1]
        bn3 = self.blocks[4][-1].conv[-1]
        bn4 = self.blocks[6][-1].conv[-1]
        return [bn1, bn2, bn3, bn4]

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        return feat_m

    def forward(self, x):

        out = self.conv1_follow(self.act(self.conv1(x)))
        # out = self.conv1(x)


        out = self.blocks[0](out)
        out = self.blocks[1](out)

        out = self.blocks[2](out)

        out = self.blocks[3](out)
        out = self.blocks[4](out)

        out = self.blocks[5](out)
        out = self.blocks[6](out)

        # out = self.conv2(out)


        out = self.avgpool(out)

        out1 = out.squeeze(3).squeeze(2)


        out3 = self.classifier(out1)


        return out3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def merge_rep_armor_mbv2(self):
        # 27592616
        self.conv1.merge_dilated_branches()
        # # 27591719
        self.conv1_follow.switch_to_deploy()
        # # # 835802
        for block in self.blocks:
            for layer in block:
                if isinstance(layer,InvertedResidual):
                    layer.merge_inverte()
        # 814602
        self.classifier.fuse()
        pass

        


def mobilenetv2_T_w(T, W, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model


def finalNet(num_classes):
    return mobilenetv2_T_w(6, 0.66, num_classes)


import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table
if __name__ == '__main__':
    f1 = torch.randn(2, 3, 32, 32)

    conv1 = finalNet(100)
    weights_dict = torch.load('/root/distill/path/teacher_model/finalNet_miniImage_lr_0.01_decay_0.0005_trial_66/finalNet_best.pth')

           
    conv1.load_state_dict(weights_dict['model'], strict=True)



    conv1.eval()
    with torch.no_grad():
        output1 = conv1(f1)
        start_time = time.time()
        for _ in range(1):
            conv1(f1)
        print(f"consume time: {time.time() - start_time}")
        total_params1 = sum(p.numel() for p in conv1.parameters())
        print('total_params_before_merge:'+str(total_params1))
        conv1.train()
        conv1.merge_rep_armor_mbv2()
        for n,p in conv1.named_parameters():
            print(n)
        conv1.eval()
        output2 = conv1(f1)
        start_time = time.time()
        for _ in range(1):
            conv1(f1)
        print(f"consume time: {time.time() - start_time}")
        total_params2 = sum(p.numel() for p in conv1.parameters())
        print('total_params_after_merge:'+str(total_params2))
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-03, atol=1e-05)
        print("convert module has been tested, and the inference results, before and after merging, are equal!")
        print(output1[1:10])
        print(output2[1:10])
        


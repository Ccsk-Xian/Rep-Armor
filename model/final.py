# # """
# # MobileNetV2 implementation used in
# # <Knowledge Distillation via Route Constrained Optimization>
# # """

# # import torch.nn as nn
# # import math

# # import torch
# # import numpy as np
# # from torch.nn import init
# # from itertools import repeat
# # from torch.nn import functional as F
# # import collections.abc as container_abcs
# # from torch._jit_internal import Optional
# # from torch.nn.parameter import Parameter
# # from torch.nn.modules.module import Module
# # __all__ = ['mobilenetv2_T_w', 'mobile_half']

# # BN = None


# # class DOConv2d(Module):
# #     """
# #        DOConv2d can be used as an alternative for torch.nn.Conv2d.
# #        The interface is similar to that of Conv2d, with one exception:
# #             1. D_mul: the depth multiplier for the over-parameterization.
# #        Note that the groups parameter switchs between DO-Conv (groups=1),
# #        DO-DConv (groups=in_channels), DO-GConv (otherwise).
# #     """
# #     __constants__ = ['stride', 'padding', 'dilation', 'groups',
# #                      'padding_mode', 'output_padding', 'in_channels',
# #                      'out_channels', 'kernel_size', 'D_mul']
# #     __annotations__ = {'bias': Optional[torch.Tensor]}

# #     def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
# #                  padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
# #         super(DOConv2d, self).__init__()

# #         kernel_size = _pair(kernel_size)
# #         stride = _pair(stride)
# #         padding = _pair(padding)
# #         dilation = _pair(dilation)

# #         if in_channels % groups != 0:
# #             raise ValueError('in_channels must be divisible by groups')
# #         if out_channels % groups != 0:
# #             raise ValueError('out_channels must be divisible by groups')
# #         valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
# #         if padding_mode not in valid_padding_modes:
# #             raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
# #                 valid_padding_modes, padding_mode))
# #         self.in_channels = in_channels
# #         self.out_channels = out_channels
# #         self.kernel_size = kernel_size
# #         self.stride = stride
# #         self.padding = padding
# #         self.dilation = dilation
# #         self.groups = groups
# #         self.padding_mode = padding_mode
# #         self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

# #         #################################### Initailization of D & W ###################################
# #         M = self.kernel_size[0]
# #         N = self.kernel_size[1]
# #         self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
# #         self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
# #         init.kaiming_uniform_(self.W, a=math.sqrt(5))

# #         if M * N > 1:
# #             self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
# #             init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
# #             self.D.data = torch.from_numpy(init_zero)

# #             eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
# #             d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
# #             if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
# #                 zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
# #                 self.d_diag = Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
# #             else:  # the case when D_mul = M * N
# #                 self.d_diag = Parameter(d_diag, requires_grad=False)
# #         ##################################################################################################

# #         if bias:
# #             self.bias = Parameter(torch.Tensor(out_channels))
# #             fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
# #             bound = 1 / math.sqrt(fan_in)
# #             init.uniform_(self.bias, -bound, bound)
# #         else:
# #             self.register_parameter('bias', None)

# #     def extra_repr(self):
# #         s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
# #              ', stride={stride}')
# #         if self.padding != (0,) * len(self.padding):
# #             s += ', padding={padding}'
# #         if self.dilation != (1,) * len(self.dilation):
# #             s += ', dilation={dilation}'
# #         if self.groups != 1:
# #             s += ', groups={groups}'
# #         if self.bias is None:
# #             s += ', bias=False'
# #         if self.padding_mode != 'zeros':
# #             s += ', padding_mode={padding_mode}'
# #         return s.format(**self.__dict__)

# #     def __setstate__(self, state):
# #         super(DOConv2d, self).__setstate__(state)
# #         if not hasattr(self, 'padding_mode'):
# #             self.padding_mode = 'zeros'

# #     def _conv_forward(self, input, weight):
# #         if self.padding_mode != 'zeros':
# #             return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
# #                             weight, self.bias, self.stride,
# #                             _pair(0), self.dilation, self.groups)
# #         return F.conv2d(input, weight, self.bias, self.stride,
# #                         self.padding, self.dilation, self.groups)

# #     def forward(self, input):
# #         M = self.kernel_size[0]
# #         N = self.kernel_size[1]
# #         DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
# #         if M * N > 1:
# #             ######################### Compute DoW #################
# #             # (input_channels, D_mul, M * N)
# #             D = self.D + self.d_diag
# #             W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

# #             # einsum outputs (out_channels // groups, in_channels, M * N),
# #             # which is reshaped to
# #             # (out_channels, in_channels // groups, M, N)
# #             DoW = torch.reshape(torch.einsum('cms,ois->oim', D, W), DoW_shape)
# #             #######################################################
# #         else:
# #             # in this case D_mul == M * N
# #             # reshape from
# #             # (out_channels, in_channels // groups, D_mul)
# #             # to
# #             # (out_channels, in_channels // groups, M, N)
# #             DoW = torch.reshape(self.W, DoW_shape)
# #         return self._conv_forward(input, DoW)


# # def _ntuple(n):
# #     def parse(x):
# #         if isinstance(x, container_abcs.Iterable):
# #             return x
# #         return tuple(repeat(x, n))

# #     return parse
# # _pair = _ntuple(2)


# # def conv1_bn(inp, oup, stride):
# #     return nn.Sequential(
# #         nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
# #         nn.BatchNorm2d(oup),
# #         nn.ReLU(inplace=True)
# #     )
# # from timm.models.vision_transformer import trunc_normal_
# # class BN_Linear(torch.nn.Sequential):
# #     def __init__(self, a, b, bias=True, std=0.02):
# #         super().__init__()
# #         self.add_module('bn', torch.nn.BatchNorm1d(a))
# #         self.add_module('drop',torch.nn.Dropout(p=0.2))
# #         self.add_module('l', torch.nn.Linear(a, b, bias=bias))
# #         trunc_normal_(self.l.weight, std=std)
# #         if bias:
# #             torch.nn.init.constant_(self.l.bias, 0)

# #     @torch.no_grad()
# #     def fuse(self):
# #         bn, l = self._modules.values()
# #         w = bn.weight / (bn.running_var + bn.eps)**0.5
# #         b = bn.bias - self.bn.running_mean * \
# #             self.bn.weight / (bn.running_var + bn.eps)**0.5
# #         w = l.weight * w[None, :]
# #         if l.bias is None:
# #             b = b @ self.l.weight.T
# #         else:
# #             b = (l.weight @ b[:, None]).view(-1) + self.l.bias
# #         m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
# #         m.weight.data.copy_(w)
# #         m.bias.data.copy_(b)
# #         return m

# # class Classfier(nn.Module):
# #     def __init__(self, dim, num_classes, distillation=True):
# #         super().__init__()
# #         self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
# #         self.distillation = distillation
# #         if distillation:
# #             self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
            
# #     def forward(self, x):
# #         if self.distillation:
# #             x = self.classifier(x), self.classifier_dist(x)
            
# #             x = (x[0] + x[1]) / 2
# #         else:
# #             x = self.classifier(x)
# #         return x

# #     @torch.no_grad()
# #     def fuse(self):
# #         classifier = self.classifier.fuse()
# #         if self.distillation:
# #             classifier_dist = self.classifier_dist.fuse()
# #             classifier.weight += classifier_dist.weight
# #             classifier.bias += classifier_dist.bias
# #             classifier.weight /= 2
# #             classifier.bias /= 2
# #             return classifier
# #         else:
# #             return classifier
        
# # class Rep_bn(nn.Module):
# #     """
# #     Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
# #     We assume the inputs to this block are (N, C, H, W)
# #     """
# #     def __init__(self, inchannels,outchannels, kernel_size, stride, bias, group):
# #         super().__init__()
# #         self.lk_origin = nn.Conv2d(inchannels, outchannels, kernel_size, stride=stride,
# #                                     padding=kernel_size//2, dilation=1, groups=group, bias=bias,
# #                                     )
# #         # self.lk_origin = DOConv2d(inchannels, outchannels, kernel_size=kernel_size, stride=stride, groups=group,padding=kernel_size//2,bias=bias)
        
       

# #         if kernel_size == 17:
# #             self.kernel_sizes = [5, 9, 3, 3, 3]  # 5 17 9 11 15
# #             self.dilates = [1, 2, 4, 5, 7]
# #         elif kernel_size == 15:
# #             self.kernel_sizes = [5, 7, 3, 3, 3] # 5 13 7 11 15
# #             self.dilates = [1, 2, 3, 5, 7]
# #         elif kernel_size == 13:
# #             self.kernel_sizes = [5, 7, 3, 3, 3]  # 5 13 7 9 11
# #             self.dilates = [1, 2, 3, 4, 5]
# #         elif kernel_size == 11:
# #             self.kernel_sizes = [5, 5, 3, 3, 3]  # 5 9 7 9 11
# #             self.dilates = [1, 2, 3, 4, 5]
# #         elif kernel_size == 9:
# #             self.kernel_sizes = [5, 5, 3, 3]  # 5 9 7 9
# #             self.dilates = [1, 2, 3, 4]
# #         elif kernel_size == 7:
# #             self.kernel_sizes = [5, 3, 3,3]  # 5 5 7
# #             self.dilates = [1, 2, 3,1]
# #         elif kernel_size == 5:
# #             self.kernel_sizes = [3, 3] # 3 5
# #             self.dilates = [1, 2]
# #         else:
# #             raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

# #         if not bias:
# #             self.origin_bn = nn.BatchNorm2d(outchannels) if inchannels==outchannels and stride ==1 else nn.Identity()
# #             for k, r in zip(self.kernel_sizes, self.dilates):
# #                 self.__setattr__('dil_conv_k{}_{}'.format(k, r),
# #                                  nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=k, stride=stride,
# #                                            padding=(r * (k - 1) + 1) // 2, dilation=r, groups=group,
# #                                            bias=bias))
# #                                 # DOConv2d(inchannels, outchannels, kernel_size=k, stride=stride, groups=group,padding=(r * (k - 1) + 1) // 2,bias=bias,dilation=r))
# #                 # self.__setattr__('conv_one_follow{}_{}'.format(k, r),
# #                 #                  nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=1, stride=1,
# #                 #                            padding=0, dilation=1, groups=group,
# #                 #                            bias=False))
# #                 self.__setattr__('dil_bn_k{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))
# #                 self.__setattr__('conv_one{}_{}'.format(k, r),
# #                                  nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=1, stride=1,
# #                                            padding=0, dilation=1, groups=1,
# #                                            bias=False))
# #                                 # DOConv2d(outchannels, outchannels, kernel_size=1, stride=1, groups=1,padding=0,bias=False,dilation=1))
# #                 self.__setattr__('bn_one{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))

# #     def forward(self, x):
# #         if not hasattr(self, 'origin_bn'):      # deploy mode
# #             return self.lk_origin(x)
# #         out = self.origin_bn(self.lk_origin(x))
# #         for k, r in zip(self.kernel_sizes, self.dilates):
# #             conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
# #             # conv1_follow = self.__getattr__('conv_one_follow{}_{}'.format(k, r))
# #             bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
# #             inner_conv1 = self.__getattr__('conv_one{}_{}'.format(k, r))
# #             bn1 = self.__getattr__('bn_one{}_{}'.format(k, r))
# #             # print(x.shape)
# #             middle = bn(conv(x))
# #             # middle = bn(conv1_follow(conv(x)))
# #             # print(middle.shape)
# #             # print(out.shape)
# #             out = out + bn1(inner_conv1(middle))
# #         return out


# # class SEBlock(nn.Module):

# #     def __init__(self, input_channels, internal_neurons):
# #         super(SEBlock, self).__init__()
# #         self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
# #         self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
# #         self.input_channels = input_channels

# #     def forward(self, inputs):
# #         x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
# #         x = self.down(x)
# #         x = F.relu(x)
# #         x = self.up(x)
# #         x = torch.sigmoid(x)
# #         x = x.view(-1, self.input_channels, 1, 1)
# #         return inputs * x

# # def conv_bn(inp, oup, stride):
# #     return nn.Sequential(
# #         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
# #         nn.BatchNorm2d(oup),
# #         nn.ReLU(inplace=True)
# #     )
# # def conv_bn_rep(in_channels, out_channels, kernel_size, stride, padding, groups=1,use_bn=True,follow1=False):
# #     result = nn.Sequential()
# #     if stride == 2 or in_channels!=out_channels:
# #         result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
# #                                                       kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
# #         # result.add_module('conv',DOConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,padding=padding,bias=False),)
# #     else:
# #         # result.add_module('conv',DOConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,padding=padding,bias=False),)
# #         result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
# #                                                   kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
# #     if follow1:
# #         result.add_module('conv1',nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
# #                                                   kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
# #     if use_bn:
# #         result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
# #     return result

# # class RecoveryBlock(nn.Module):

# #     def __init__(self, in_channels, out_channels, kernel_size,
# #                  stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,use_act=True,use_bn=True,use_identity=True):
# #         super(RecoveryBlock, self).__init__()
# #         self.deploy = deploy
# #         self.groups = groups
# #         self.in_channels = in_channels
# #         self.use_act = use_act

       
# #         if self.use_act:
# #             self.nonlinearity = nn.ReLU()
# #         else:
# #             self.nonlinearity = nn.Identity()

# #         if use_se:
# #             #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
# #             self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
# #         else:
# #             self.se = nn.Identity()

# #         if deploy:
# #             self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
# #                                       padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

# #         else:
# #             # self.rbr_identity_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
# #             # self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 and use_identity else None
# #             # self.dense_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
# #             self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow1=True)
# #             # self.dense_weight_2 = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
# #             self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow1=False)
# #             # self.rbr_dense_3 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
# #             # self.rbr_dense_4 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
# #             # self.rbr_dense_5 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
            
# #             # self.rbr_1x1 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups,use_bn=use_bn)
# #             # self.rbr_1x1_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
# #             # torch.nn.init.constant_(self.dense_weight,0.5)
# #             # torch.nn.init.constant_(self.dense_weight_2,0.5)
# #             # torch.nn.init.constant_(self.rbr_1x1_weight,1)
# #             # torch.nn.init.constant_(self.rbr_identity_weight,1)
# #             # print('RepVGG Block, identity = ', self.rbr_identity)


# #     def forward(self, inputs):
# #         if hasattr(self, 'rbr_reparam'):
# #             return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

# #         # if self.rbr_identity is None:
# #         #     id_out = 0
# #         # else:
# #         #     id_out = self.rbr_identity(inputs)
# #         return self.nonlinearity(self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs))/2))
# #         # return self.nonlinearity(self.se((self.dense_weight*self.rbr_dense(inputs) + self.rbr_1x1_weight*self.rbr_1x1(inputs))/2+id_out))


# #     #   Optional. This may improve the accuracy and facilitates quantization in some cases.
# #     #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
# #     #   2.  Use like this.
# #     #       loss = criterion(....)
# #     #       for every RecoveryBlock blk:
# #     #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
# #     #       optimizer.zero_grad()
# #     #       loss.backward()
# #     def get_custom_L2(self):
# #         K3 = self.rbr_dense.conv.weight
# #         K1 = self.rbr_1x1.conv.weight
# #         t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
# #         t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

# #         l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
# #         eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
# #         l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
# #         return l2_loss_eq_kernel + l2_loss_circle



# # #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
# # #   You can get the equivalent kernel and bias at any time and do whatever you want,
# #     #   for example, apply some penalties or constraints during training, just like you do to the other models.
# # #   May be useful for quantization or pruning.
# #     def get_equivalent_kernel_bias(self):
# #         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
# #         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
# #         kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
# #         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

# #     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
# #         if kernel1x1 is None:
# #             return 0
# #         else:
# #             return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

# #     def _fuse_bn_tensor(self, branch):
# #         if branch is None:
# #             return 0, 0
# #         if isinstance(branch, nn.Sequential):
# #             kernel = branch.conv.weight
# #             running_mean = branch.bn.running_mean
# #             running_var = branch.bn.running_var
# #             gamma = branch.bn.weight
# #             beta = branch.bn.bias
# #             eps = branch.bn.eps
# #         else:
# #             assert isinstance(branch, nn.BatchNorm2d)
# #             if not hasattr(self, 'id_tensor'):
# #                 input_dim = self.in_channels // self.groups
# #                 kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
# #                 for i in range(self.in_channels):
# #                     kernel_value[i, i % input_dim, 1, 1] = 1
# #                 self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
# #             kernel = self.id_tensor
# #             running_mean = branch.running_mean
# #             running_var = branch.running_var
# #             gamma = branch.weight
# #             beta = branch.bias
# #             eps = branch.eps
# #         std = (running_var + eps).sqrt()
# #         t = (gamma / std).reshape(-1, 1, 1, 1)
# #         return kernel * t, beta - running_mean * gamma / std

# #     def switch_to_deploy(self):
# #         if hasattr(self, 'rbr_reparam'):
# #             return
# #         kernel, bias = self.get_equivalent_kernel_bias()
# #         self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
# #                                      kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
# #                                      padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        
        
# #         self.rbr_reparam.weight.data = kernel
# #         self.rbr_reparam.bias.data = bias
# #         self.__delattr__('rbr_dense')
# #         self.__delattr__('rbr_1x1')
# #         if hasattr(self, 'rbr_identity'):
# #             self.__delattr__('rbr_identity')
# #         if hasattr(self, 'id_tensor'):
# #             self.__delattr__('id_tensor')
# #         self.deploy = True



# # def conv_bn(inp, oup, stride):
# #     return nn.Sequential(
# #         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
# #         nn.BatchNorm2d(oup),
# #         nn.ReLU(inplace=True)
# #     )


# # def conv_1x1_bn(inp, oup):
# #     return nn.Sequential(
# #         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
# #         nn.BatchNorm2d(oup),
# #         nn.ReLU(inplace=True)
# #     )


# # class InvertedResidual(nn.Module):
# #     def __init__(self, inp, oup, stride, expand_ratio):
# #         super(InvertedResidual, self).__init__()
# #         self.blockname = None

# #         self.stride = stride
# #         assert stride in [1, 2]

# #         self.use_res_connect = self.stride == 1 and inp == oup
# #         # if stride == 2:
# #         #     self.pw = RecoveryBlock(in_channels=inp, out_channels=inp * expand_ratio, groups=1, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False,use_bn=False)
# #         # else:
# #         #     self.pw = RecoveryBlock(in_channels=inp, out_channels=inp * expand_ratio, groups=1, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False)
# #         self.dw = RecoveryBlock(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio, groups=inp * expand_ratio, kernel_size=3, stride=stride, padding=1, deploy=False, use_se=False,use_identity=True)
# #         # self.pw_linear = nn.Sequential(nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
# #         #         nn.BatchNorm2d(oup),)
        
# #         self.pw = nn.Sequential(
# #             # pw
# #             nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
# #             # DOConv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, groups=1,padding=0,bias=False),
# #             nn.BatchNorm2d(inp * expand_ratio),
# #             nn.ReLU(inplace=True),)
# #             # dw
# #             # nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
# #             # nn.BatchNorm2d(inp * expand_ratio),
# #             # nn.ReLU(inplace=True),
# #             # pw-linear
# #         self.conv = nn.Sequential(    
# #             # DOConv2d(inp * expand_ratio,oup, kernel_size=1, stride=1, groups=1,padding=0,bias=False),
# #             nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
# #             nn.BatchNorm2d(oup),
# #         )
# #         self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

# #     def forward(self, x):
# #         t = x
# #         if self.use_res_connect:
# #             # return t+self.conv(x)
# #             return t + self.conv(self.dw(self.pw(x)))
# #         else:
# #             # return(self.conv(x))
# #             return self.conv(self.dw(self.pw(x)))


# # class MobileNetV2(nn.Module):
# #     """mobilenetV2"""
# #     def __init__(self, T,
# #                  feature_dim,
# #                  input_size=32,
# #                  width_mult=1.,
# #                  remove_avg=False):
# #         super(MobileNetV2, self).__init__()
# #         self.remove_avg = remove_avg

# #         # setting of inverted residual blocks  cifar 第二行s为1
# #         self.interverted_residual_setting = [
# #             # t, c, n, s
# #             [1, 16, 1, 1],
# #             [T, 24, 2, 2],
# #             [T, 32, 3, 2],
# #             [T, 64, 4, 2],
# #             [T, 96, 3, 1],
# #             [T, 160, 3, 2],
# #             [T, 320, 1, 1],
# #         ]


# #         # building first layer
# #         assert input_size % 32 == 0
# #         input_channel = int(32 * width_mult)
# #         # self.conv1 = conv_bn(3, input_channel, 2)
# #         self.conv1 = Rep_bn(inchannels=3,outchannels=3,kernel_size=7,stride=2,bias=False,group=1)
# #         self.act = nn.ReLU(inplace=True)
# #         self.conv1_follow = conv1_bn(3,input_channel,1)
# #         # building inverted residual blocks
# #         self.blocks = nn.ModuleList([])
# #         for t, c, n, s in self.interverted_residual_setting:
# #             output_channel = int(c * width_mult)
# #             layers = []
# #             strides = [s] + [1] * (n - 1)
# #             for stride in strides:
# #                 layers.append(
# #                     InvertedResidual(input_channel, output_channel, stride, t)
# #                 )
# #                 input_channel = output_channel
# #             self.blocks.append(nn.Sequential(*layers))

# #         # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else int(c*width_mult)*4
# #         # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
# #         # self.conv2 = conv_1x1_bn(input_channel, self.last_channel)
# #         # print(self.last_channel)
# #         # self.conv2 = RecoveryBlock(in_channels=input_channel, out_channels=self.last_channel, groups=1, kernel_size=1, stride=1, padding=0, deploy=False,use_act=False, use_se=False)
# #         # print(input_channel)
        
# #         # self.conv2_rep = RecoveryBlock(in_channels=input_channel , out_channels=self.last_channel , kernel_size=3, stride=1, padding=1, deploy=False, use_se=False,use_act=False, groups=input_channel)
# #         # self.actfinal = nn.ReLU(inplace=True)

# #         self.classifier = Classfier(output_channel, feature_dim, True)
# #         # building classifier
# #         # self.classifier = nn.Sequential(
# #         #     nn.Dropout(0.2),
# #         #     nn.Linear(self.last_channel, feature_dim),
# #         # )

# #         # H = input_size // (32//2)
# #         # self.avgpool = nn.AvgPool2d(H, ceil_mode=True)
# #         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

# #         self._initialize_weights()
# #         print(T, width_mult)

# #     def get_bn_before_relu(self):
# #         bn1 = self.blocks[1][-1].conv[-1]
# #         bn2 = self.blocks[2][-1].conv[-1]
# #         bn3 = self.blocks[4][-1].conv[-1]
# #         bn4 = self.blocks[6][-1].conv[-1]
# #         return [bn1, bn2, bn3, bn4]

# #     def get_feat_modules(self):
# #         feat_m = nn.ModuleList([])
# #         feat_m.append(self.conv1)
# #         feat_m.append(self.blocks)
# #         return feat_m

# #     def forward(self, x, is_feat=False, preact=False):

# #         out = self.conv1_follow(self.act(self.conv1(x)))
# #         # out = self.conv1(x)
# #         f0 = out

# #         out = self.blocks[0](out)
# #         out = self.blocks[1](out)
# #         f1 = out
# #         out = self.blocks[2](out)
# #         f2 = out
# #         out = self.blocks[3](out)
# #         out = self.blocks[4](out)
# #         f3 = out
# #         out = self.blocks[5](out)
# #         out = self.blocks[6](out)
# #         f4 = out

# #         # out = self.conv2(out)

# #         if not self.remove_avg:
# #             out = self.avgpool(out)
# #         out = out.view(out.size(0), -1)
# #         f5 = out
# #         out = self.classifier(out)

# #         if is_feat:
# #             return [f0, f1, f2, f3, f4, f5], out
# #         else:
# #             return out

# #     def _initialize_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
# #                 m.weight.data.normal_(0, math.sqrt(2. / n))
# #                 if m.bias is not None:
# #                     m.bias.data.zero_()
# #             elif isinstance(m, nn.BatchNorm2d):
# #                 m.weight.data.fill_(1)
# #                 m.bias.data.zero_()
# #             elif isinstance(m, nn.Linear):
# #                 n = m.weight.size(1)
# #                 m.weight.data.normal_(0, 0.01)
# #                 m.bias.data.zero_()


# # def mobilenetv2_T_w(T, W, feature_dim=100):
# #     model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
# #     return model


# # def RepTinynet23(num_classes):
# #     return mobilenetv2_T_w(6, 0.66, num_classes)

# # from fvcore.nn import FlopCountAnalysis, parameter_count_table
# # if __name__ == '__main__':
# #     x = torch.randn(2, 3, 224, 224)

# #     net = RepTinynet23(100)

# #     feats, logit = net(x, is_feat=True, preact=True)
# #     for f in feats:
# #         print(f.shape, f.min().item())
# #     print(logit.shape)
# #     flops = FlopCountAnalysis(net,x)
# #     params = parameter_count_table(net)
# #     print(flops.total())
# #     print(params)
# #     for m in net.get_bn_before_relu():
# #         if isinstance(m, nn.BatchNorm2d):
# #             print('pass')
# #         else:
# #             print('warning')



# """
# MobileNetV2 implementation used in
# <Knowledge Distillation via Route Constrained Optimization>
# """
# import torch.nn as nn
# import math

# import torch
# import numpy as np
# from torch.nn import init
# from itertools import repeat
# from torch.nn import functional as F
# import collections.abc as container_abcs
# from torch._jit_internal import Optional
# from torch.nn.parameter import Parameter
# from torch.nn.modules.module import Module
# __all__ = ['mobilenetv2_T_w', 'mobile_half']

# BN = None
# class RepTowerBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,expand_rate = 3,neck_rate=0.5):
#         super(RepTowerBlock, self).__init__()
#         self.deploy = deploy
#         self.groups = groups
#         self.in_channels = in_channels

     
#         # 不能引入bias，所以只能是identity
#         self.skip = nn.Identity() if in_channels==out_channels else nn.BatchNorm2d(in_channels)
#         # self.skip = nn.Sequential(
            

#         # )
        
#         if deploy:
#             self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
#                                       padding=0, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
#         else:
#             self.needle = nn.Sequential(
#                 # nn.Conv2d(in_channels,in_channels,1,1,0,bias=False,groups=groups),
#                 nn.BatchNorm2d(in_channels),
#                 nn.Conv2d(in_channels,int(in_channels*expand_rate),1,1,0,bias=False,groups=groups),
#                 # nn.BatchNorm2d(int(in_channels*expand_rate)),
#                 nn.Conv2d(int(in_channels*expand_rate),int(in_channels*expand_rate),1,1,0,bias=False,groups=groups),
#                 # nn.BatchNorm2d(int(in_channels*expand_rate)),
#                 nn.Conv2d(int(in_channels*expand_rate),out_channels,1,1,0,bias=False,groups=groups),
#                 # nn.BatchNorm2d(out_channels),
#             )
#             # if groups==1:
#             #     self.down_needle = nn.Sequential(
#             #         nn.Conv2d(in_channels,int(in_channels*neck_rate),1,1,0,bias=False,groups=int(groups*neck_rate) if groups>1 else 1),
#             #         # nn.Conv2d(int(in_channels*neck_rate),int(in_channels*neck_rate),1,1,0,bias=False,groups=int(groups*neck_rate) if groups>1 else 1),
#             #         nn.Conv2d(int(in_channels*neck_rate),out_channels,1,1,0,bias=False,groups=int(groups*neck_rate) if groups>1 else 1),
#             #     )
#             # self.rbr_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups,bias=False)


#     def forward(self, inputs):
#         if hasattr(self, 'rbr_reparam'):
#             return self.rbr_reparam(inputs)
#         # if self.groups==1:
#         #     return (self.needle(inputs)+self.down_needle(inputs))/2+self.skip(inputs)
#         # else:
#         return self.needle(inputs)+self.skip(inputs)
#         # return self.needle(inputs)
# class DOConv2d(Module):
#     """
#        DOConv2d can be used as an alternative for torch.nn.Conv2d.
#        The interface is similar to that of Conv2d, with one exception:
#             1. D_mul: the depth multiplier for the over-parameterization.
#        Note that the groups parameter switchs between DO-Conv (groups=1),
#        DO-DConv (groups=in_channels), DO-GConv (otherwise).
#     """
#     __constants__ = ['stride', 'padding', 'dilation', 'groups',
#                      'padding_mode', 'output_padding', 'in_channels',
#                      'out_channels', 'kernel_size', 'D_mul']
#     __annotations__ = {'bias': Optional[torch.Tensor]}

#     def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
#         super(DOConv2d, self).__init__()

#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)

#         if in_channels % groups != 0:
#             raise ValueError('in_channels must be divisible by groups')
#         if out_channels % groups != 0:
#             raise ValueError('out_channels must be divisible by groups')
#         valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
#         if padding_mode not in valid_padding_modes:
#             raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
#                 valid_padding_modes, padding_mode))
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.padding_mode = padding_mode
#         self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

#         #################################### Initailization of D & W ###################################
#         M = self.kernel_size[0]
#         N = self.kernel_size[1]
#         self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
#         self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
#         init.kaiming_uniform_(self.W, a=math.sqrt(5))

#         if M * N > 1:
#             self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
#             init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
#             self.D.data = torch.from_numpy(init_zero)

#             eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
#             d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
#             if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
#                 zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
#                 self.d_diag = Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
#             else:  # the case when D_mul = M * N
#                 self.d_diag = Parameter(d_diag, requires_grad=False)
#         ##################################################################################################

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)
#         else:
#             self.register_parameter('bias', None)

#     def extra_repr(self):
#         s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
#              ', stride={stride}')
#         if self.padding != (0,) * len(self.padding):
#             s += ', padding={padding}'
#         if self.dilation != (1,) * len(self.dilation):
#             s += ', dilation={dilation}'
#         if self.groups != 1:
#             s += ', groups={groups}'
#         if self.bias is None:
#             s += ', bias=False'
#         if self.padding_mode != 'zeros':
#             s += ', padding_mode={padding_mode}'
#         return s.format(**self.__dict__)

#     def __setstate__(self, state):
#         super(DOConv2d, self).__setstate__(state)
#         if not hasattr(self, 'padding_mode'):
#             self.padding_mode = 'zeros'

#     def _conv_forward(self, input, weight):
#         if self.padding_mode != 'zeros':
#             return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
#                             weight, self.bias, self.stride,
#                             _pair(0), self.dilation, self.groups)
#         return F.conv2d(input, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#     def forward(self, input):
#         M = self.kernel_size[0]
#         N = self.kernel_size[1]
#         DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
#         if M * N > 1:
#             ######################### Compute DoW #################
#             # (input_channels, D_mul, M * N)
#             D = self.D + self.d_diag
#             W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

#             # einsum outputs (out_channels // groups, in_channels, M * N),
#             # which is reshaped to
#             # (out_channels, in_channels // groups, M, N)
#             DoW = torch.reshape(torch.einsum('cms,ois->oim', D, W), DoW_shape)
#             #######################################################
#         else:
#             # in this case D_mul == M * N
#             # reshape from
#             # (out_channels, in_channels // groups, D_mul)
#             # to
#             # (out_channels, in_channels // groups, M, N)
#             DoW = torch.reshape(self.W, DoW_shape)
#         return self._conv_forward(input, DoW)


# def _ntuple(n):
#     def parse(x):
#         if isinstance(x, container_abcs.Iterable):
#             return x
#         return tuple(repeat(x, n))

#     return parse
# _pair = _ntuple(2)

# __all__ = ['mobilenetv2_T_w', 'mobile_half']

# BN = None

# def conv1_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)
#     )
# from timm.models.vision_transformer import trunc_normal_
# class BN_Linear(torch.nn.Sequential):
#     def __init__(self, a, b, bias=True, std=0.02):
#         super().__init__()
#         # self.add_module('bn', torch.nn.BatchNorm1d(a))
#         self.add_module('drop',torch.nn.Dropout(p=0.2))
#         self.add_module('l', torch.nn.Linear(a, b, bias=bias))
#         trunc_normal_(self.l.weight, std=std)
#         if bias:
#             torch.nn.init.constant_(self.l.bias, 0)

#     @torch.no_grad()
#     def fuse(self):
#         bn, l = self._modules.values()
#         w = bn.weight / (bn.running_var + bn.eps)**0.5
#         b = bn.bias - self.bn.running_mean * \
#             self.bn.weight / (bn.running_var + bn.eps)**0.5
#         w = l.weight * w[None, :]
#         if l.bias is None:
#             b = b @ self.l.weight.T
#         else:
#             b = (l.weight @ b[:, None]).view(-1) + self.l.bias
#         m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
#         m.weight.data.copy_(w)
#         m.bias.data.copy_(b)
#         return m

# class Classfier(nn.Module):
#     def __init__(self, dim, num_classes, distillation=True):
#         super().__init__()
#         self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
#         self.distillation = distillation
#         if distillation:
#             self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
            
#     def forward(self, x):
#         if self.distillation:
#             x = self.classifier(x), self.classifier_dist(x)
            
#             x = (x[0] + x[1]) / 2
#         else:
#             x = self.classifier(x)
#         return x

#     @torch.no_grad()
#     def fuse(self):
#         classifier = self.classifier.fuse()
#         if self.distillation:
#             classifier_dist = self.classifier_dist.fuse()
#             classifier.weight += classifier_dist.weight
#             classifier.bias += classifier_dist.bias
#             classifier.weight /= 2
#             classifier.bias /= 2
#             return classifier
#         else:
#             return classifier
        
# class Rep_bn(nn.Module):
#     """
#     Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
#     We assume the inputs to this block are (N, C, H, W)
#     """
#     def __init__(self, inchannels,outchannels, kernel_size, stride, bias, group):
#         super().__init__()
#         self.lk_origin = nn.Conv2d(inchannels, outchannels, kernel_size, stride=stride,
#                                     padding=kernel_size//2, dilation=1, groups=group, bias=bias,
#                                     )
#         self.origin_1x1 = RepTowerBlock(outchannels,outchannels,groups=group,expand_rate=4)
#         # self.lk_origin = DOConv2d(inchannels, outchannels, kernel_size=kernel_size, stride=stride, groups=group,padding=kernel_size//2,bias=bias)
       

#         if kernel_size == 17:
#             self.kernel_sizes = [5, 9, 3, 3, 3]  # 5 17 9 11 15
#             self.dilates = [1, 2, 4, 5, 7]
#         elif kernel_size == 15:
#             self.kernel_sizes = [5, 7, 3, 3, 3] # 5 13 7 11 15
#             self.dilates = [1, 2, 3, 5, 7]
#         elif kernel_size == 13:
#             self.kernel_sizes = [5, 7, 3, 3, 3]  # 5 13 7 9 11
#             self.dilates = [1, 2, 3, 4, 5]
#         elif kernel_size == 11:
#             self.kernel_sizes = [5, 5, 3, 3, 3]  # 5 9 7 9 11
#             self.dilates = [1, 2, 3, 4, 5]
#         elif kernel_size == 9:
#             self.kernel_sizes = [5, 5, 3, 3]  # 5 9 7 9
#             self.dilates = [1, 2, 3, 4]
#         elif kernel_size == 7:
#             self.kernel_sizes = [5, 3, 3,3]  # 5 5 7
#             self.dilates = [1, 2, 3,1]
#         elif kernel_size == 5:
#             self.kernel_sizes = [3, 3] # 3 5
#             self.dilates = [1, 2]
#         elif kernel_size == 3:
#             self.kernel_sizes = [3] # 3 
#             self.dilates = [1]
#         # else:
#         #     raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

#         if not bias:
#             self.origin_bn = nn.BatchNorm2d(outchannels) 
#             for k, r in zip(self.kernel_sizes, self.dilates):
#                 self.__setattr__('dil_conv_k{}_{}'.format(k, r),
#                                  nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=k, stride=stride,
#                                            padding=(r * (k - 1) + 1) // 2, dilation=r, groups=group,
#                                            bias=bias)
#                                 # DOConv2d(inchannels, outchannels, kernel_size=k, stride=stride, groups=group,padding=(r * (k - 1) + 1) // 2,bias=bias,dilation=r)
#                                            )
#                 self.__setattr__('dil_bn_k{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))
#                 self.__setattr__('conv_one{}_{}'.format(k, r),
#                                 RepTowerBlock(outchannels,outchannels,groups=group,expand_rate=4))
#                                 # DOConv2d(outchannels, outchannels, kernel_size=1, stride=1, groups=1,padding=0,bias=False,dilation=1))
#                                 #  nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=1, stride=1,
#                                 #            padding=0, dilation=1, groups=1,
#                                 #            bias=False))
#                                 # # DOConv2d(outchannels, outchannels, kernel_size=1, stride=1, groups=1,padding=0,bias=False,dilation=1))
#                 # self.__setattr__('bn_one{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))

#     def forward(self, x):
#         if not hasattr(self, 'origin_bn'):      # deploy mode
#             return self.lk_origin(x)
#         out = self.origin_1x1(self.origin_bn(self.lk_origin(x)))
#         # out = self.origin_bn(self.lk_origin(x))
#         for k, r in zip(self.kernel_sizes, self.dilates):
#             conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
#             bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
#             inner_conv1 = self.__getattr__('conv_one{}_{}'.format(k, r))
#             # bn1 = self.__getattr__('bn_one{}_{}'.format(k, r))
#             # print(x.shape)
#             # middle = bn(conv(x))
#             # print(middle.shape)
#             # print(out.shape)
#             # out = out + bn1(inner_conv1(middle))
#             out = out +bn(inner_conv1(conv(x)))
#             # out = out+bn(conv(x))
#         return out
#         # return bn1(inner_conv1(out))


# class SEBlock(nn.Module):

#     def __init__(self, input_channels, internal_neurons):
#         super(SEBlock, self).__init__()
#         self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
#         self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
#         self.input_channels = input_channels

#     def forward(self, inputs):
#         x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
#         x = self.down(x)
#         x = F.relu(x)
#         x = self.up(x)
#         x = torch.sigmoid(x)
#         x = x.view(-1, self.input_channels, 1, 1)
#         return inputs * x

# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)
#     )
# def conv_bn_rep(in_channels, out_channels, kernel_size, stride, padding, groups=1,use_bn=True,follow=False):
#     result = nn.Sequential()
#     if stride == 2 or in_channels!=out_channels:
#         result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                                       kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
#         # result.add_module('conv',DOConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,padding=padding,bias=False),)
#     else:
#         result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                                       kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
#         # result.add_module('conv',DOConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,padding=padding,bias=False),)
#     if follow:
#         result.add_module('conv1',RepTowerBlock(in_channels=out_channels,out_channels=out_channels,groups=groups))
#     if use_bn:
#         result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
#     return result

# class RecoveryBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,use_act=True,use_bn=True,use_identity=True):
#         super(RecoveryBlock, self).__init__()
#         self.deploy = deploy
#         self.groups = groups
#         self.in_channels = in_channels
#         self.use_act = use_act

       
#         if self.use_act:
#             self.nonlinearity = nn.ReLU()
#         else:
#             self.nonlinearity = nn.Identity()

#         if use_se:
#             #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
#             self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
#         else:
#             self.se = nn.Identity()

#         if deploy:
#             self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                                       padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

#         else:
#             # self.rbr_identity_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
#             # self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 and use_identity else None
#             # self.dense_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
#             self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True)
#             # self.dense_weight_2 = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
#             self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True)
#             # self.rbr_dense_3 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
#             # self.rbr_dense_4 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
#             # self.rbr_dense_5 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
            
#             # self.rbr_1x1 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups,use_bn=use_bn)
#             # self.rbr_1x1_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
#             # torch.nn.init.constant_(self.dense_weight,0.1)
#             # torch.nn.init.constant_(self.dense_weight_2,1.0)
#             # torch.nn.init.constant_(self.rbr_1x1_weight,0.1)
#             # torch.nn.init.constant_(self.rbr_identity_weight,0.1)
#             # print('RepVGG Block, identity = ', self.rbr_identity)


#     def forward(self, inputs):
#         if hasattr(self, 'rbr_reparam'):
#             return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

#         # if self.rbr_identity is None:
#         #     id_out = 0
#         # else:
#         #     id_out = self.rbr_identity(inputs)

#         return self.nonlinearity(self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs))/2))
#         # return self.nonlinearity(self.se(self.dense_weight*self.rbr_dense(inputs) + self.rbr_1x1_weight*self.rbr_1x1(inputs)+id_out))


#     #   Optional. This may improve the accuracy and facilitates quantization in some cases.
#     #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
#     #   2.  Use like this.
#     #       loss = criterion(....)
#     #       for every RecoveryBlock blk:
#     #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
#     #       optimizer.zero_grad()
#     #       loss.backward()
#     def get_custom_L2(self):
#         K3 = self.rbr_dense.conv.weight
#         K1 = self.rbr_1x1.conv.weight
#         t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
#         t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

#         l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
#         eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
#         l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
#         return l2_loss_eq_kernel + l2_loss_circle



# #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
# #   You can get the equivalent kernel and bias at any time and do whatever you want,
#     #   for example, apply some penalties or constraints during training, just like you do to the other models.
# #   May be useful for quantization or pruning.
#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
#         kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

#     def _fuse_bn_tensor(self, branch):
#         if branch is None:
#             return 0, 0
#         if isinstance(branch, nn.Sequential):
#             kernel = branch.conv.weight
#             running_mean = branch.bn.running_mean
#             running_var = branch.bn.running_var
#             gamma = branch.bn.weight
#             beta = branch.bn.bias
#             eps = branch.bn.eps
#         else:
#             assert isinstance(branch, nn.BatchNorm2d)
#             if not hasattr(self, 'id_tensor'):
#                 input_dim = self.in_channels // self.groups
#                 kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
#                 for i in range(self.in_channels):
#                     kernel_value[i, i % input_dim, 1, 1] = 1
#                 self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
#             kernel = self.id_tensor
#             running_mean = branch.running_mean
#             running_var = branch.running_var
#             gamma = branch.weight
#             beta = branch.bias
#             eps = branch.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std

#     def switch_to_deploy(self):
#         if hasattr(self, 'rbr_reparam'):
#             return
#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
#                                      kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
#                                      padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        
        
#         self.rbr_reparam.weight.data = kernel
#         self.rbr_reparam.bias.data = bias
#         self.__delattr__('rbr_dense')
#         self.__delattr__('rbr_1x1')
#         if hasattr(self, 'rbr_identity'):
#             self.__delattr__('rbr_identity')
#         if hasattr(self, 'id_tensor'):
#             self.__delattr__('id_tensor')
#         self.deploy = True



# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)
#     )


# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)
#     )


# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.blockname = None

#         self.stride = stride
#         assert stride in [1, 2]

#         self.use_res_connect = self.stride == 1 and inp == oup
#         # if stride == 2:
#         #     self.pw = RecoveryBlock(in_channels=inp, out_channels=inp * expand_ratio, groups=1, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False,use_bn=False)
#         # else:
#         #     self.pw = RecoveryBlock(in_channels=inp, out_channels=inp * expand_ratio, groups=1, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False)
#         self.dw = RecoveryBlock(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio, groups=inp * expand_ratio, kernel_size=3, stride=stride, padding=1, deploy=False, use_se=False,use_identity=True)
#         # self.pw_linear = nn.Sequential(nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
#         #         nn.BatchNorm2d(oup),)
        
#         if stride==2:
#             self.pw = nn.Sequential(
#                 # pw
#                 nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
#                 # DOConv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, groups=1,padding=0,bias=False),
#                 # RepTowerBlock(inp * expand_ratio,inp * expand_ratio,expand_rate=2),
#                 nn.BatchNorm2d(inp * expand_ratio),
#                 nn.ReLU(inplace=True),)
#         else:
#             self.pw = nn.Sequential(
#                 # pw
#                 nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
#                 # DOConv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, groups=1,padding=0,bias=False),
#                 RepTowerBlock(inp * expand_ratio,inp * expand_ratio,expand_rate=4),
#                 nn.BatchNorm2d(inp * expand_ratio),
#                 nn.ReLU(inplace=True),)
#             # dw
#         # self.dw = nn.Sequential(nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
#         #     nn.BatchNorm2d(inp * expand_ratio),
#         #     nn.ReLU(inplace=True),)
#             # pw-linear
#         self.conv = nn.Sequential(    
#             # DOConv2d(inp * expand_ratio,oup, kernel_size=1, stride=1, groups=1,padding=0,bias=False),
#             nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(oup),
#         )
#         self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

#     def forward(self, x):
#         t = x
#         if self.use_res_connect:
#             # return t+self.conv(x)
#             return t + self.conv(self.dw(self.pw(x)))
#         else:
#             # return(self.conv(x))
#             return self.conv(self.dw(self.pw(x)))


# class MobileNetV2(nn.Module):
#     """mobilenetV2"""
#     def __init__(self, T,
#                  feature_dim,
#                  input_size=32,
#                  width_mult=1.,
#                  remove_avg=False):
#         super(MobileNetV2, self).__init__()
#         self.remove_avg = remove_avg

#         # setting of inverted residual blocks  cifar 第二行s为1
#         self.interverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [T, 24, 2, 2],
#             [T, 32, 3, 2],
#             [T, 64, 4, 2],
#             [T, 96, 3, 1],
#             [T, 160, 3, 2],
#             [T, 320, 1, 1],
#         ]


#         # building first layer
#         assert input_size % 32 == 0
#         input_channel = int(32 * width_mult)
#         # self.conv1 = conv_bn(3, input_channel, 2)
#         self.conv1 = Rep_bn(inchannels=3,outchannels=3,kernel_size=7,stride=1,bias=False,group=1)
#         self.act = nn.ReLU(inplace=True)
#         # self.norm1 = nn.BatchNorm2d(3)
#         # self.conv1_follow = conv1_bn(3,input_channel,1)
#         self.conv1_follow = conv_bn(3, input_channel, 2)
#         # building inverted residual blocks
#         self.blocks = nn.ModuleList([])
#         for t, c, n, s in self.interverted_residual_setting:
#             output_channel = int(c * width_mult)
#             layers = []
#             strides = [s] + [1] * (n - 1)
#             for stride in strides:
#                 layers.append(
#                     InvertedResidual(input_channel, output_channel, stride, t)
#                 )
#                 input_channel = output_channel
#             self.blocks.append(nn.Sequential(*layers))

#         # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else int(c*width_mult)*4
#         # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
#         # self.conv2 = conv_1x1_bn(input_channel, self.last_channel)
#         # print(self.last_channel)
#         # self.conv2 = RecoveryBlock(in_channels=input_channel, out_channels=self.last_channel, groups=1, kernel_size=1, stride=1, padding=0, deploy=False,use_act=False, use_se=False)
#         # print(input_channel)
        
#         # self.conv2_rep = RecoveryBlock(in_channels=input_channel , out_channels=self.last_channel , kernel_size=3, stride=1, padding=1, deploy=False, use_se=False,use_act=False, groups=input_channel)
#         # self.actfinal = nn.ReLU(inplace=True)

#         self.classifier = Classfier(output_channel, feature_dim, True)
#         # building classifier
#         # self.classifier = nn.Sequential(
#         #     nn.Dropout(0.2),
#         #     nn.Linear(self.last_channel, feature_dim),
#         # )

#         # H = input_size // (32//2)
#         # self.avgpool = nn.AvgPool2d(H, ceil_mode=True)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         self._initialize_weights()
#         print(T, width_mult)

#     def get_bn_before_relu(self):
#         bn1 = self.blocks[1][-1].conv[-1]
#         bn2 = self.blocks[2][-1].conv[-1]
#         bn3 = self.blocks[4][-1].conv[-1]
#         bn4 = self.blocks[6][-1].conv[-1]
#         return [bn1, bn2, bn3, bn4]

#     def get_feat_modules(self):
#         feat_m = nn.ModuleList([])
#         feat_m.append(self.conv1)
#         feat_m.append(self.blocks)
#         return feat_m

#     def forward(self, x, is_feat=False, preact=False):

#         out = self.conv1_follow(self.act(self.conv1(x)))
#         # out = self.conv1(x)
#         f0 = out

#         out = self.blocks[0](out)
#         out = self.blocks[1](out)
#         f1 = out
#         out = self.blocks[2](out)
#         f2 = out
#         out = self.blocks[3](out)
#         out = self.blocks[4](out)
#         f3 = out
#         out = self.blocks[5](out)
#         out = self.blocks[6](out)
#         f4 = out

#         # out = self.conv2(out)

#         if not self.remove_avg:
#             out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         f5 = out
#         out = self.classifier(out)

#         if is_feat:
#             return [f0, f1, f2, f3, f4, f5], out
#         else:
#             return out

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()


# def mobilenetv2_T_w(T, W, feature_dim=100):
#     model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
#     return model


# def finalNet(num_classes):
#     # return mobilenetv2_T_w(6, 0.66, num_classes)
#     return mobilenetv2_T_w(2, 1.12, num_classes)

# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# if __name__ == '__main__':
#     x = torch.randn(2, 3, 32, 32)

#     net = finalNet(100)

#     feats, logit = net(x, is_feat=True, preact=True)
#     for f in feats:
#         print(f.shape, f.min().item())
#     print(logit.shape)
#     flops = FlopCountAnalysis(net,x)
#     params = parameter_count_table(net)
#     print(flops.total())
#     print(params)
#     for m in net.get_bn_before_relu():
#         if isinstance(m, nn.BatchNorm2d):
#             print('pass')
#         else:
#             print('warning')




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
            # if in_channels>3:
            #     self.lowneedle = nn.Sequential(
            #         nn.Conv2d(in_channels,int(in_channels//4),1,1,0,bias=False,groups=groups),
            #             # nn.Conv2d(int(in_channels*expand_rate),int(in_channels*expand_rate),1,1,0,bias=False,groups=groups),
            #             nn.BatchNorm2d(int(in_channels//4)),
            #             nn.Conv2d(int(in_channels//4),out_channels,1,1,0,bias=False,groups=groups),
            #     )
            # else:
            #     self.lowneedle=nn.Identity()
            # if groups==1:
            #     self.down_needle = nn.Sequential(
            #         nn.Conv2d(in_channels,int(in_channels*neck_rate),1,1,0,bias=False,groups=int(groups*neck_rate) if groups>1 else 1),
            #         # nn.Conv2d(int(in_channels*neck_rate),int(in_channels*neck_rate),1,1,0,bias=False,groups=int(groups*neck_rate) if groups>1 else 1),
            #         nn.Conv2d(int(in_channels*neck_rate),out_channels,1,1,0,bias=False,groups=int(groups*neck_rate) if groups>1 else 1),
            #     )
            # self.rbr_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups,bias=False)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.rbr_reparam(inputs)
        # if self.groups==1:
        #     return (self.needle(inputs)+self.down_needle(inputs))/2+self.skip(inputs)
        # else:
        return self.needle(inputs)+self.skip(inputs)
        # return inputs
        # return self.needle(inputs)
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

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
            
    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            
            x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier
        
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
        # self.lk_origin = DOConv2d(inchannels, outchannels, kernel_size=kernel_size, stride=stride, groups=group,padding=kernel_size//2,bias=bias)
       

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
        # else:
        #     raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not bias:
            self.origin_bn = nn.BatchNorm2d(outchannels) 
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=k, stride=stride,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=group,
                                           bias=bias)
                                # DOConv2d(inchannels, outchannels, kernel_size=k, stride=stride, groups=group,padding=(r * (k - 1) + 1) // 2,bias=bias,dilation=r)
                                           )
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))
                self.__setattr__('conv_one{}_{}'.format(k, r),
                                RepTowerBlock(inchannels,inchannels,groups=group,expand_rate=4))
                                # DOConv2d(outchannels, outchannels, kernel_size=1, stride=1, groups=1,padding=0,bias=False,dilation=1))
                                #  nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=1, stride=1,
                                #            padding=0, dilation=1, groups=1,
                                #            bias=False))
                                # # DOConv2d(outchannels, outchannels, kernel_size=1, stride=1, groups=1,padding=0,bias=False,dilation=1))
                # self.__setattr__('bn_one{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_1x1(self.origin_bn(self.lk_origin(x)))
        # out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            inner_conv1 = self.__getattr__('conv_one{}_{}'.format(k, r))
            # bn1 = self.__getattr__('bn_one{}_{}'.format(k, r))
            # print(x.shape)
            # middle = bn(conv(x))
            # print(middle.shape)
            # print(out.shape)
            # out = out + bn1(inner_conv1(middle))
            # out = out +inner_conv1(bn(conv(x)))
            out = out +bn(conv(inner_conv1(x)))
            # out = out+bn(conv(x))
        # return out/math.sqrt(len(self.kernel_sizes))
        return out/math.sqrt(len(self.kernel_sizes))
        # return out 


class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

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

# class RecoveryBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
#         super(RecoveryBlock, self).__init__()
#         self.deploy = deploy
#         self.groups = groups
#         self.in_channels = in_channels

#         # assert kernel_size == 3
#         # assert padding == 1

#         # padding_11 = padding - kernel_size // 2

#         self.nonlinearity = nn.ReLU()

#         if use_se:
#             #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
#             self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
#         else:
#             self.se = nn.Identity()

#         if deploy:
#             self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                                       padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

#         else:
#             self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
#             self.rbr_dense = conv_bn1(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
#             self.rbr_1x1 = conv_bn1(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
#             print('RepVGG Block, identity = ', self.rbr_identity)


#     def forward(self, inputs):
#         if hasattr(self, 'rbr_reparam'):
#             return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

#         if self.rbr_identity is None:
#             id_out = 0
#         else:
#             id_out = self.rbr_identity(inputs)

#         return self.nonlinearity(self.se((self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

class RecoveryBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,use_act=True,use_bn=True,use_identity=True,count=True):
        super(RecoveryBlock, self).__init__()
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
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            # self.rbr_identity_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
            # self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 and use_identity else None
            # self.dense_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
            self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=3)
            # self.dense_weight_2 = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
            self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=2)
            self.rbr_dense_3 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=1)
            self.rbr_dense_4 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False)
            self.rbr_dense_5 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False)
            
            # self.rbr_1x1 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups,use_bn=use_bn)
            # self.rbr_1x1_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
            # torch.nn.init.constant_(self.dense_weight,0.1)
            # torch.nn.init.constant_(self.dense_weight_2,1.0)
            # torch.nn.init.constant_(self.rbr_1x1_weight,0.1)
            # torch.nn.init.constant_(self.rbr_identity_weight,0.1)
            # print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        # if self.rbr_identity is None:
        #     id_out = 0
        # else:
        #     id_out = self.rbr_identity(inputs)

        # return self.nonlinearity(self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs))/2))
        return self.nonlinearity(self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs)+self.rbr_dense_3(inputs)+self.rbr_dense_4(inputs)+self.rbr_dense_5(inputs))/5))
        # return self.nonlinearity(self.se(self.dense_weight*self.rbr_dense(inputs) + self.rbr_1x1_weight*self.rbr_1x1(inputs)+id_out))


    #   Optional. This may improve the accuracy and facilitates quantization in some cases.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RecoveryBlock blk:
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



# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)
#     )


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
        # RecoveryBlock are used to enhance 1x1 convs' capacity
        self.pw = RecoveryBlock(in_channels=inp, out_channels=inp * expand_ratio, groups=1, kernel_size=1, stride=1, padding=0,count=count)
        # RecoveryBlock exhibts the similar performance than DBB, but DBB is slightly excellent for convs with kernel size larger than 3
        self.dw = conv_bn_relu(inp*expand_ratio,inp*expand_ratio,3,stride,1,1,inp*expand_ratio)
        
        self.conv = nn.Sequential(    
            # DOConv2d(inp * expand_ratio,oup, kernel_size=1, stride=1, groups=1,padding=0,bias=False),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x
        if self.use_res_connect:
            # return t+self.conv(x)
            return t + self.conv(self.dw(self.pw(x)))
        else:
            # return(self.conv(x))
            # return self.conv(self.dw(self.pw(x)))
            return self.conv(self.dw(self.pw(x)))


class MobileNetV2(nn.Module):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 remove_avg=False):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg

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
        # self.conv1 = conv_bn(3, input_channel, 2)

        # 1. large conv layer is implemented by the Uni block (dilated structural re-parameterization + RepTower + Ensemble Modification)
        self.conv1 = Rep_bn(inchannels=3,outchannels=3,kernel_size=7,stride=1,bias=False,group=1)

        self.act = nn.ReLU(inplace=True)
        # self.norm1 = nn.BatchNorm2d(3)
        # self.conv1_follow = conv1_bn(3,input_channel,1)
        # 2.the original Stem layer is enhanced by the DBB block
        self.conv1_follow = conv_bn_relu(3,input_channel,3,2,1)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s,count in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                # 3. recovery blocks are used in InvertedResidual
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t,count)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else int(c*width_mult)*4
        # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        # self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

        

        # 4. The simplest classifier with ensemble structural re-parameterization, plus dropout layers are excepted to bring more diversity for branches
        self.classifier = Classfier(output_channel, feature_dim, True)
        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, feature_dim),
        # )

        # H = input_size // (32//2)
        # self.avgpool = nn.AvgPool2d(H, ceil_mode=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

    def forward(self, x, is_feat=False, preact=False):

        out = self.conv1_follow(self.act(self.conv1(x)))
        # out = self.conv1(x)
        f0 = out

        out = self.blocks[0](out)
        out = self.blocks[1](out)
        f1 = out
        out = self.blocks[2](out)
        f2 = out
        out = self.blocks[3](out)
        out = self.blocks[4](out)
        f3 = out
        out = self.blocks[5](out)
        out = self.blocks[6](out)
        f4 = out

        # out = self.conv2(out)

        if not self.remove_avg:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.classifier(out)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out
        else:
            return out

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


def mobilenetv2_T_w(T, W, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model


def finalNet(num_classes):
    return mobilenetv2_T_w(6, 0.66, num_classes)

from fvcore.nn import FlopCountAnalysis, parameter_count_table
if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)

    net = finalNet(100)

    feats, logit = net(x, is_feat=True, preact=True)
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
    flops = FlopCountAnalysis(net,x)
    params = parameter_count_table(net)
    print(flops.total())
    print(params)
    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')

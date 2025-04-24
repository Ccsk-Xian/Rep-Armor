# """
# MobileNetV2 implementation used in
# <Knowledge Distillation via Route Constrained Optimization>
# """

# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# __all__ = ['mobilenetv2_T_w', 'mobile_half']

# BN = None

# def conv1_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
# from timm.models.vision_transformer import trunc_normal_
# class BN_Linear(torch.nn.Sequential):
#     def __init__(self, a, b, bias=True, std=0.02):
#         super().__init__()
#         # self.add_module('bn', torch.nn.BatchNorm1d(a))
#         # self.add_module('drop',torch.nn.Dropout(p=0.2))
#         self.add_module('l', torch.nn.Linear(a, b, bias=bias))
#         # trunc_normal_(self.l.weight, std=std)
#         # if bias:
#         #     torch.nn.init.constant_(self.l.bias, 0)

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
#         else:
#             raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

#         if not bias:
#             self.origin_bn = nn.BatchNorm2d(outchannels) if inchannels==outchannels and stride ==1 else nn.Identity()
#             for k, r in zip(self.kernel_sizes, self.dilates):
#                 self.__setattr__('dil_conv_k{}_{}'.format(k, r),
#                                  nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=k, stride=stride,
#                                            padding=(r * (k - 1) + 1) // 2, dilation=r, groups=group,
#                                            bias=bias))
#                 self.__setattr__('dil_bn_k{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))
#                 self.__setattr__('conv_one{}_{}'.format(k, r),
#                                  nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=1, stride=1,
#                                            padding=0, dilation=1, groups=1,
#                                            bias=False))
#                 self.__setattr__('bn_one{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))

#     def forward(self, x):
#         if not hasattr(self, 'origin_bn'):      # deploy mode
#             return self.lk_origin(x)
#         out = self.origin_bn(self.lk_origin(x))
#         for k, r in zip(self.kernel_sizes, self.dilates):
#             conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
#             bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
#             inner_conv1 = self.__getattr__('conv_one{}_{}'.format(k, r))
#             bn1 = self.__getattr__('bn_one{}_{}'.format(k, r))
#             # print(x.shape)
#             middle = bn(conv(x))
#             # print(middle.shape)
#             # print(out.shape)
#             out = out + bn1(inner_conv1(middle))
#         return out


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
#         nn.ReLU6(inplace=True)
#     )
# def conv_bn_rep(in_channels, out_channels, kernel_size, stride, padding, groups=1,use_bn=True):
#     result = nn.Sequential()
#     result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                                   kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
#     if use_bn:
#         result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
#     return result

# class RepVGGBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,use_act=True,use_bn=True,use_identity=True):
#         super(RepVGGBlock, self).__init__()
#         self.deploy = deploy
#         self.groups = groups
#         self.in_channels = in_channels
#         self.use_act = use_act

       
#         if self.use_act:
#             # self.nonlinearity = nn.ReLU()
#             self.nonlinearity = nn.LeakyReLU(0.1)
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
#             self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 and use_identity else None
#             # self.dense_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
#             self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
#             # self.dense_weight_2 = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
#             self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
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

#         if self.rbr_identity is None:
#             id_out = 0
#         else:
#             id_out = self.rbr_identity(inputs)
#         # return self.nonlinearity(self.se(self.rbr_dense(inputs)*self.dense_weight+self.rbr_dense_2(inputs)*self.dense_weight_2))
#         return self.nonlinearity(self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs))/2 +id_out))


#     #   Optional. This may improve the accuracy and facilitates quantization in some cases.
#     #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
#     #   2.  Use like this.
#     #       loss = criterion(....)
#     #       for every RepVGGBlock blk:
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
#         nn.ReLU6(inplace=True)
#     )


# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )


# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio,kernel,use_skip):
#         super(InvertedResidual, self).__init__()
#         self.blockname = None

#         self.stride = stride
#         assert stride in [1, 2]


#         self.use_res_connect = self.stride == 1 and inp == oup

#         self.conv = nn.Sequential(
#             # pw
#             nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(inp * expand_ratio),
#             nn.ReLU(inplace=True),
#             # dw
#             nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel//2, groups=inp * expand_ratio, bias=False),
#             # DOConv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, groups=inp * expand_ratio,padding=1,bias=False),
#             nn.BatchNorm2d(inp * expand_ratio),
#             nn.ReLU(inplace=True),
#             # pw-linear
#             nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(oup),
#         )
        
        
        
#         self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

#     def forward(self, x):
#         t = x
#         if self.use_res_connect:
#             return t+self.conv(x)
#             # return t + self.conv(self.dw(self.pw(x)))
#         else:
#             return(self.conv(x))
#             # return self.conv(self.dw(self.pw(x)))


# class MobileNetV2(nn.Module):
#     """mobilenetV2"""
#     def __init__(self, T,
#                  feature_dim,
#                  input_size=16,
#                  width_mult=1.,
#                  remove_avg=False):
#         super(MobileNetV2, self).__init__()
#         self.remove_avg = remove_avg

#         # setting of inverted residual blocks  cifar 第二行s为1
#         self.interverted_residual_setting = [
#             # t, c, k, s
#             [1, 8, 3, 1],
#             [3, 16, 5, 2],
#             [6, 16, 7, 1],
#             [5, 16, 3, 1],
#             [5, 16, 5, 1],
#             [5, 24, 3, 2],
#             [6, 24, 7, 1],
#             [6, 24, 5, 1],
#             [4, 40, 7, 2],
#             [5, 40, 5, 1],
#             [5, 48, 3, 1],
#             [5, 48, 5, 1],
#             [4, 48, 3, 1],
#             [6, 96, 5, 2],
#             [4, 96, 5, 1],
#             [3, 96, 5, 1],
#             [4, 96, 3, 1],
#             [5, 160, 5, 1],


#             # [1, 16, 1, 1],
#             # [T, 24, 2, 2],
#             # [T, 32, 3, 2],
#             # [T, 64, 4, 2],
#             # [T, 96, 3, 1],
#             # [T, 160, 3, 2],
#             # [T, 320, 1, 1],
#         ]


#         # building first layer
#         assert input_size % 16 == 0
#         input_channel = int(16 * width_mult)
#         self.conv1 = conv_bn(3, input_channel, 2)
#         # self.conv1 = Rep_bn(inchannels=3,outchannels=3,kernel_size=7,stride=2,bias=False,group=1)
#         # self.act = nn.ReLU6(inplace=True)
#         # self.conv1_follow = conv1_bn(3,input_channel,1)
#         # building inverted residual blocks
#         self.blocks = nn.ModuleList([])
#         for t, c, k, s in self.interverted_residual_setting:
#             use_skip = c>16
#             output_channel = int(c * width_mult)
#             layers = []
#             # strides = [s] + [1] * (n - 1)
#             # for stride in strides:
#             layers.append(
#                 InvertedResidual(input_channel, output_channel, s, t,k,use_skip)
#             )
#             input_channel = output_channel
#             self.blocks.append(nn.Sequential(*layers))

#         # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else int(c*width_mult)*4
#         # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
#         # self.conv2 = conv_1x1_bn(input_channel, self.last_channel)
#         # print(self.last_channel)
#         # self.conv2 = RepVGGBlock(in_channels=input_channel, out_channels=self.last_channel, groups=1, kernel_size=1, stride=1, padding=0, deploy=False,use_act=False, use_se=False)
#         # print(input_channel)
        
#         # self.conv2_rep = RepVGGBlock(in_channels=input_channel , out_channels=self.last_channel , kernel_size=3, stride=1, padding=1, deploy=False, use_se=False,use_act=False, groups=input_channel)
#         # self.actfinal = nn.ReLU6(inplace=True)

#         self.classifier = Classfier(output_channel, feature_dim, False)
#         # building classifier
#         # self.classifier = nn.Sequential(
#         #     nn.Dropout(0.2),
#         #     nn.Linear(self.last_channel, feature_dim),
#         # )

#         # H = input_size // (32//2)
#         # self.avgpool = nn.AvgPool2d(H, ceil_mode=True)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         # self._initialize_weights()
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

#         # out = self.conv1_follow(self.act(self.conv1(x)))
#         out = self.conv1(x)
#         # f0 = out

#         # out = self.blocks[0](out)
#         # out = self.blocks[1](out)
#         # f1 = out
#         # out = self.blocks[2](out)
#         # f2 = out
#         # out = self.blocks[3](out)
#         # out = self.blocks[4](out)
#         # f3 = out
#         # out = self.blocks[5](out)
#         # out = self.blocks[6](out)
#         # f4 = out
#         for f in self.blocks:
#             out = f(out)
#         # out = self.conv2(out)

#         if not self.remove_avg:
#             out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         f5 = out
#         out = self.classifier(out)

#         # if is_feat:
#             # return [f0, f1, f2, f3, f4, f5], out
#         # else:
#         return out

#     # def _initialize_weights(self):
#     #     for m in self.modules():
#     #         if isinstance(m, nn.Conv2d):
#     #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#     #             m.weight.data.normal_(0, math.sqrt(2. / n))
#     #             if m.bias is not None:
#     #                 m.bias.data.zero_()
#     #         elif isinstance(m, nn.BatchNorm2d):
#     #             m.weight.data.fill_(1)
#     #             m.bias.data.zero_()
#     #         elif isinstance(m, nn.Linear):
#     #             n = m.weight.size(1)
#     #             m.weight.data.normal_(0, 0.01)
#     #             m.bias.data.zero_()


# def mobilenetv2_T_w(T, W, feature_dim=100):
#     model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
#     return model


# def MCUnetv1(num_classes):
#     return mobilenetv2_T_w(6, 1, num_classes)

# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# if __name__ == '__main__':
#     x = torch.randn(2, 3, 224, 224)

#     net = MCUnetv1(100)

#     logit = net(x)
#     # for f in feats:
#     #     print(f.shape, f.min().item())
#     print(logit.shape)
#     flops = FlopCountAnalysis(net,x)
#     params = parameter_count_table(net)
#     print(flops.total())
#     print(params)
#     # for m in net.get_bn_before_relu():
#     #     if isinstance(m, nn.BatchNorm2d):
#     #         print('pass')
#     #     else:
#     #         print('warning')


"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
__all__ = ['mobilenetv2_T_w', 'mobile_half']

BN = None

def conv1_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
from timm.models.vision_transformer import trunc_normal_
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('drop',torch.nn.Dropout(p=0.2))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        # trunc_normal_(self.l.weight, std=std)
        # if bias:
        #     torch.nn.init.constant_(self.l.bias, 0)

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
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not bias:
            self.origin_bn = nn.BatchNorm2d(outchannels) if inchannels==outchannels and stride ==1 else nn.Identity()
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=k, stride=stride,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=group,
                                           bias=bias))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))
                self.__setattr__('conv_one{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=1, stride=1,
                                           padding=0, dilation=1, groups=1,
                                           bias=False))
                self.__setattr__('bn_one{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            inner_conv1 = self.__getattr__('conv_one{}_{}'.format(k, r))
            bn1 = self.__getattr__('bn_one{}_{}'.format(k, r))
            # print(x.shape)
            middle = bn(conv(x))
            # print(middle.shape)
            # print(out.shape)
            out = out + bn1(inner_conv1(middle))
        return out


class Repvgg_Rep_bn(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, inchannels,outchannels, kernel_size, stride, bias, group):
        super().__init__()
        self.lk_origin = nn.Conv2d(inchannels, outchannels, kernel_size, stride=stride,
                                    padding=kernel_size//2, dilation=1, groups=group, bias=bias,
                                    )
       

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
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not bias:
            self.origin_bn = nn.BatchNorm2d(outchannels) if inchannels==outchannels and stride ==1 else nn.Identity()
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=k, stride=stride,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=group,
                                           bias=bias))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))
                # self.__setattr__('conv_one{}_{}'.format(k, r),
                #                  nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=1, stride=1,
                #                            padding=0, dilation=1, groups=1,
                #                            bias=False))
                # self.__setattr__('bn_one{}_{}'.format(k, r), nn.BatchNorm2d(outchannels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            # inner_conv1 = self.__getattr__('conv_one{}_{}'.format(k, r))
            # bn1 = self.__getattr__('bn_one{}_{}'.format(k, r))
            # print(x.shape)
            
            # print(middle.shape)
            # print(out.shape)
            # middle = bn(conv(x))
            # out = out + bn1(inner_conv1(middle))
            out = bn(conv(x))+out
        return out/(len(self.kernel_sizes)+1)

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
        nn.ReLU6(inplace=True)
    )
def conv_bn_rep(in_channels, out_channels, kernel_size, stride, padding, groups=1,use_bn=True):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    if use_bn:
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,use_act=True,use_bn=True,use_identity=True):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.use_act = use_act

       
        if self.use_act:
            self.nonlinearity = nn.ReLU6()
            # self.nonlinearity = nn.LeakyReLU(0.1)
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
            # self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 and use_identity else None
            # self.dense_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
            if kernel_size>=5:
                self.rbr_dense = Repvgg_Rep_bn(inchannels=in_channels,outchannels=out_channels,kernel_size=kernel_size,stride=stride,group=groups,bias=False)
                self.rbr_dense_2 = Repvgg_Rep_bn(inchannels=in_channels,outchannels=out_channels,kernel_size=kernel_size,stride=stride,group=groups,bias=False)
                # self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
            # self.dense_weight_2 = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
                # self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
            else:
                self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
            # self.dense_weight_2 = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
                self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)

            # self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
            # self.dense_weight_2 = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
            # self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
            # self.rbr_dense_3 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
            # self.rbr_dense_4 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
            # self.rbr_dense_5 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn)
            
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
        #     return self.nonlinearity(self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs))/2 +id_out))
        # else:
        #     id_out = self.rbr_identity(inputs)
        #     return self.nonlinearity((self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs))/2 +id_out)))
        return self.nonlinearity((self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs))/2)))
        # return self.nonlinearity(self.se(self.rbr_dense(inputs)*self.dense_weight+self.rbr_dense_2(inputs)*self.dense_weight_2))
        


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



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,kernel,use_skip):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]


        self.use_res_connect = self.stride == 1 and inp == oup

        # if stride == 2:
        #     self.pw = RepVGGBlock(in_channels=inp, out_channels=inp * expand_ratio, groups=1, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False,use_bn=False)
        # else:
        #     self.pw = RepVGGBlock(in_channels=inp, out_channels=inp * expand_ratio, groups=1, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False)
        self.dw = RepVGGBlock(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio, groups=inp * expand_ratio, kernel_size=kernel, stride=stride, padding=kernel//2, deploy=False, use_se=False,use_identity=True)
        # self.pw_linear = nn.Sequential(nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
        #         nn.BatchNorm2d(oup),)
        
        self.pw = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),)
            # dw
        # self.dw = nn.Sequential(nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel//2, groups=inp * expand_ratio, bias=False),
        #     nn.BatchNorm2d(inp * expand_ratio),
        #     nn.ReLU6(inplace=True),)
            # pw-linear
        self.conv = nn.Sequential(    nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )


        # self.conv = nn.Sequential(
        #     # pw
        #     nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(inp * expand_ratio),
        #     nn.ReLU6(inplace=True),
        #     # dw
        #     nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel//2, groups=inp * expand_ratio, bias=False),
        #     # DOConv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, groups=inp * expand_ratio,padding=1,bias=False),
        #     nn.BatchNorm2d(inp * expand_ratio),
        #     nn.ReLU6(inplace=True),
        #     # pw-linear
        #     nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(oup),
        # )
        
        
        
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x
        if self.use_res_connect:
            # return t+self.conv(x)
            return t + self.conv(self.dw(self.pw(x)))
        else:
            # return(self.conv(x))
            return self.conv(self.dw(self.pw(x)))


class MobileNetV2(nn.Module):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=16,
                 width_mult=1.,
                 remove_avg=False):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg

        # setting of inverted residual blocks  cifar 第二行s为1
        self.interverted_residual_setting = [
            # t, c, k, s
            [1, 8, 3, 1],
            [3, 16, 5, 2],
            [6, 16, 7, 1],
            [5, 16, 3, 1],
            [5, 16, 5, 1],
            [5, 24, 3, 2],
            [6, 24, 7, 1],
            [6, 24, 5, 1],
            [4, 40, 7, 2],
            [5, 40, 5, 1],
            [5, 48, 3, 1],
            [5, 48, 5, 1],
            [4, 48, 3, 1],
            [6, 96, 5, 2],
            [4, 96, 5, 1],
            [3, 96, 5, 1],
            [4, 96, 3, 1],
            [5, 160, 5, 1],


            # [1, 16, 1, 1],
            # [T, 24, 2, 2],
            # [T, 32, 3, 2],
            # [T, 64, 4, 2],
            # [T, 96, 3, 1],
            # [T, 160, 3, 2],
            # [T, 320, 1, 1],
        ]


        # building first layer
        assert input_size % 16 == 0
        input_channel = int(16 * width_mult)
        # self.conv1 = conv_bn(3, input_channel, 2)
        self.conv1 = Rep_bn(inchannels=3,outchannels=3,kernel_size=7,stride=2,bias=False,group=1)
        self.act = nn.ReLU6(inplace=True)
        self.conv1_follow = conv1_bn(3,input_channel,1)
        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, k, s in self.interverted_residual_setting:
            use_skip = c>16
            output_channel = int(c * width_mult)
            layers = []
            # strides = [s] + [1] * (n - 1)
            # for stride in strides:
            layers.append(
                InvertedResidual(input_channel, output_channel, s, t,k,use_skip)
            )
            input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else int(c*width_mult)*4
        # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        # self.conv2 = conv_1x1_bn(input_channel, self.last_channel)
        # print(self.last_channel)
        # self.conv2 = RepVGGBlock(in_channels=input_channel, out_channels=self.last_channel, groups=1, kernel_size=1, stride=1, padding=0, deploy=False,use_act=False, use_se=False)
        # print(input_channel)
        
        # self.conv2_rep = RepVGGBlock(in_channels=input_channel , out_channels=self.last_channel , kernel_size=3, stride=1, padding=1, deploy=False, use_se=False,use_act=False, groups=input_channel)
        # self.actfinal = nn.ReLU6(inplace=True)

        self.classifier = Classfier(output_channel, feature_dim, True)
        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, feature_dim),
        # )

        # H = input_size // (32//2)
        # self.avgpool = nn.AvgPool2d(H, ceil_mode=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self._initialize_weights()
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
        # f0 = out

        # out = self.blocks[0](out)
        # out = self.blocks[1](out)
        # f1 = out
        # out = self.blocks[2](out)
        # f2 = out
        # out = self.blocks[3](out)
        # out = self.blocks[4](out)
        # f3 = out
        # out = self.blocks[5](out)
        # out = self.blocks[6](out)
        # f4 = out
        for f in self.blocks:
            out = f(out)
        # out = self.conv2(out)

        if not self.remove_avg:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.classifier(out)

        # if is_feat:
            # return [f0, f1, f2, f3, f4, f5], out
        # else:
        return out

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()


def mobilenetv2_T_w(T, W, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model


def MCUnetv1(num_classes):
    return mobilenetv2_T_w(6, 1.2, num_classes)

from fvcore.nn import FlopCountAnalysis, parameter_count_table
if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)

    net = MCUnetv1(100)

    logit = net(x)
    # for f in feats:
    #     print(f.shape, f.min().item())
    print(logit.shape)
    flops = FlopCountAnalysis(net,x)
    params = parameter_count_table(net)
    print(flops.total())
    print(params)
    # for m in net.get_bn_before_relu():
    #     if isinstance(m, nn.BatchNorm2d):
    #         print('pass')
    #     else:
    #         print('warning')



# """
# MobileNetV2 implementation used in
# <Knowledge Distillation via Route Constrained Optimization>
# """

# import torch
# import torch.nn as nn
# import math

# __all__ = ['mobilenetv2_T_w', 'mobile_half']

# BN = None


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

#         self.conv = nn.Sequential(
#             # pw
#             nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(inp * expand_ratio),
#             nn.ReLU(inplace=True),
#             # dw
#             nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
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
#             return t + self.conv(x)
#         else:
#             return self.conv(x)


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
#         # self.interverted_residual_setting = [
#         #     # t, c, n, s
#         #     [1, 16, 1, 1],
#         #     [T, 24, 2, 1],
#         #     [T, 32, 3, 2],
#         #     [T, 64, 4, 1],
#         #     [T, 96, 3, 1],
#         #     [T, 160, 3, 2],
#         #     [T, 320, 1, 1],
#         # ]
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
#         self.conv1 = conv_bn(3, input_channel, 2)

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

#         self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
#         # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else int(c*width_mult)*4
#         self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

#         # building classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, feature_dim),
#         )

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
#         f00=x
#         out = self.conv1(x)
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

#         out = self.conv2(out)

#         if not self.remove_avg:
#             out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         f5 = out
#         out = self.classifier(out)

#         if is_feat:
#             return [f00,f0, f1, f2, f3, f4, f5], out
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


# def mobile_half(num_classes):
#     return mobilenetv2_T_w(6, 0.66, num_classes)
# from thop import profile
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# import tracemalloc
# if __name__ == '__main__':
#     x = torch.randn(1, 3, 224, 224)

#     net = mobile_half(100)
#     tracemalloc.start()
#     with torch.no_grad():
#         feats, logit = net(x, is_feat=True, preact=True)
#     snapshot = tracemalloc.take_snapshot()
#     top_stats = snapshot.statistics('lineno')

#     # 找到峰值内存占用
#     peak_memory = max(stat.size / (1024 * 1024) for stat in top_stats)
#     print(f"Peak memory usage: {peak_memory:.2f} MB")

#     # 停止内存分配跟踪
#     tracemalloc.stop()
#     for f in feats:
#         print(f.shape, f.min().item())
#     print(logit.shape)
#     flops = FlopCountAnalysis(net,x)
#     params = parameter_count_table(net)
#     print(flops.total())
#     print(params)
#     flops, params = profile(net, (x,))
#     print('flops: ', flops, 'params: ', params)
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
__all__ = ['mobilenetv2_T_w', 'mobile_half']

BN = None


def create_gaussian_mask(KH,KW, noise = True, level=0.1, sigma=1.0,mask_type=1,normalization=False,weight=[],circle=True,offset=[]):
    # Create a grid of indices
    
    # grid_x = torch.arange(start=(-(KW - 1) / 2.), end=((KW - 1) / 2.))
    grid_x = torch.linspace(-(KW - 1) / 2., (KW - 1) / 2., KW).to(sigma.device)
    # print(grid_x)
    grid_y = torch.linspace(-(KH - 1) / 2., (KH - 1) / 2., KH).to(sigma.device)
    # grid_y = torch.arange(-(KH - 1) / 2., (KH - 1) / 2., KH)
    # print(grid_x)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y,indexing='xy')
    if noise:
        noise_level = level
        grid_x = grid_x + noise_level * torch.randn(*grid_x.shape).to(sigma.device)
        grid_y = grid_y + noise_level * torch.randn(*grid_y.shape).to(sigma.device)
    
    # if circle:
    #     if mask_type==2:
    #         sigma = sigma.view(sigma.shape[0],sigma.shape[1],1,1)
    #         grid_x=grid_x.expand(sigma.shape[1],KH,KW)
    #         grid_y=grid_y.expand(sigma.shape[1],KH,KW)
    #     else:
    #         sigma = sigma.view(sigma.shape[0],1,1)
        
    #         grid_x=grid_x.expand(sigma.shape[0],KH,KW)
    #         grid_y=grid_y.expand(sigma.shape[0],KH,KW)
    # else:
    #     if mask_type==2:
    #         sigma = sigma.view(2,sigma.shape[1],sigma.shape[2],1,1)
    #         grid_x=grid_x.expand(sigma.shape[2],KH,KW)
    #         grid_y=grid_y.expand(sigma.shape[2],KH,KW)
    #     else:
    #         sigma = sigma.view(2,sigma.shape[1],1,1)
    #         grid_x=grid_x.expand(sigma.shape[1],KH,KW)
    #         grid_y=grid_y.expand(sigma.shape[1],KH,KW)

    if mask_type ==1:
        
        # print(grid_x)
        # 添加随机噪声到网格坐标
        # if noise:
        #     noise_level = level  # 噪声水平，可以根据需要调整
        #     grid_x = grid_x + noise_level * torch.randn(*grid_x.shape).to(sigma.device)
        #     grid_y = grid_y + noise_level * torch.randn(*grid_y.shape).to(sigma.device)
        # print(grid)
    # grid_z = np.zeros_like(grid_x)
    
    # Calculate the Gaussian function
        if circle:
            
            if len(offset)>1:
                # gaussian = torch.exp(-0.5 * ((grid_x-offset[0].reshape(grid_x.shape))**2 + (grid_y-offset[1].reshape(grid_x.shape))**2) / sigma**2)
                gaussian = torch.exp(-0.5 * ((grid_x-offset[0])**2 + (grid_y-offset[1])**2) / sigma**2)
            else:
                gaussian = torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma**2)
        else:
            
            if len(offset)>1:
                # gaussian = torch.exp(-0.5 * ((grid_x-offset[0].reshape(grid_x.shape))**2/sigma[0]**2 + (grid_y-offset[1].reshape(grid_x.shape))**2/sigma[1]**2))

                gaussian = torch.exp(-0.5 * ((grid_x-offset[0])**2/sigma[0]**2 + (grid_y-offset[1])**2/sigma[1]**2))
            else:
                gaussian = torch.exp(-0.5 * (grid_x**2/sigma[0]**2 + grid_y**2/sigma[1]**2))
        
    # print(gaussian.shape)
    # Normalize the Gaussian mask so that the maximum value is 1
    
    if mask_type == 2:
        b = torch.softmax(weight, dim=0)
        i=torch.tensor([-1,-1,-1,0,0,0,1,1,1]).to(sigma.device)
        o=torch.tensor([-1,0,1,-1,0,1,-1,0,1]).to(sigma.device)
        i = i.view(KH*KW,1,1)
        o = o.view(KH*KW,1,1)
        grid_x=grid_x.expand(KH*KW,-1,-1)
        grid_y=grid_y.expand(KH*KW,-1,-1)
        b=b.view(KH*KW,1,1)
        if circle==True:
            sigma = sigma.view(KH*KW,1,1)

            if len(offset)>1:
                offset = offset.view(2,KH*KW,1,1)
                # gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2 + (grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2) / sigma[i*KH+o]**2)*b[i*KH+o]
                gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0])**2 + (grid_y-o-offset[1])**2) / sigma**2)*b
            else:
                # print(grid_x.shape)
                # print(sigma[i*KH+o].shape)
                gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma**2)*b
        else:
            sigma = sigma.view(2,KH*KW,1,1)
            if len(offset)>1:
                offset = offset.view(2,KH*KW,1,1)
                # gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
                gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0])**2)/sigma[0]**2 + ((grid_y-o-offset[1])**2)/sigma[1]**2))*b
            else:
                gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0]**2 + ((grid_y-o)**2)/sigma[1]**2))*b
        gaussian = gaussian.sum(dim=0)
        # gaussian = gaussian/(KH*KW)

        # for i,x in enumerate([-1,0,1]):
            # for o,y in enumerate([-1,0,1]):
        # for i in [-1,0,1]:
        #     for o in [-1,0,1]:
        #         if i==-1 and o==-1:
        # #             if circle==True:
        # #                 gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)
        # #             else:
        # #                 gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))
        # #         else:
        # #             if circle==True:
        # #                 gaussian = gaussian + torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)
        # #             else:
        # #                 gaussian = gaussian+ torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))
        # # gaussian = gaussian/(KH*KW)
        #             if circle==True:
        #                 if len(offset)>1:
        #                     # gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2 + (grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #                     gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o])**2 + (grid_y-o-offset[1][i*KH+o])**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #                 else:
        #                     # print(grid_x.shape)
        #                     # print(sigma[i*KH+o].shape)
        #                     gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #             else:
        #                 if len(offset)>1:
        #                     # gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #                     gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o])**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o])**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #                 else:
        #                     gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #         else:
        #             if circle==True:
        #                 if len(offset)>1:
        #                     # gaussian = gaussian+torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2 + (grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #                     gaussian = gaussian+torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o])**2 + (grid_y-o-offset[1][i*KH+o])**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #                 else:
        #                     gaussian = gaussian+torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #             else:
        #                 if len(offset)>1:
        #                     # gaussian = gaussian+torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #                     gaussian = gaussian+torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o])**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o])**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #                 else:
        #                     gaussian = gaussian+torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]

                # if i==-1 and o==-1:
                #     if circle==True:
                #         gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
                #     else:
                #         gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
                # else:
                #     if circle==True:
                #         gaussian = gaussian + torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma[i*KH+o]**2)*b[i*KH+o]
                #     else:
                #         gaussian = gaussian+ torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0][i*KH+o]**2 + ((grid_y-o)**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        # gaussian = gaussian/9.0
            
        # all_x = torch.zeros((KH*KW,KH,KW),dtype=torch.float32).to(weight.device)
        # all_y = torch.zeros((KH*KW,KH,KW),dtype=torch.float32).to(weight.device)
        # for i in range(KH):
        #     # grid_x = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        #     # grid_y_pre = np.linspace(-(KH-1-i),i,KH)
        #     grid_y_pre = torch.linspace(-(KH-1-i),i,KH)
        #     # print(grid_x)
        #     for o in range(KW):
        #         grid_x = torch.linspace(-(KW-1-o),o,KW)
        #         grid_x, grid_y = torch.meshgrid(grid_x, grid_y_pre,indexing='xy')
        #         # print(grid_x)
        #         all_x[i*KH+o]=grid_x
        #         all_y[i*KH+o]=grid_y
        # # weight = weight
        # # gaussian = torch.zeros((KH,KW),requires_grad=True).to(weight.device)
        # # grid_x = grid_x.to(weight.device)
        # # grid_y = grid_y.to(weight.device)
        # b = torch.softmax(weight, dim=0)
        # for i in range(KH*KW):
        #     if noise:
        #         noise_level = level
        #         grid_x = all_x[i] + noise_level * torch.randn(*grid_x.shape).to(b.device)
        #         grid_y = all_y[i] + noise_level * torch.randn(*grid_y.shape).to(b.device)
        #     # print(i)
        #     # print(type(grid_x))
        #     # print(grid_y.device)
        #     # print(weight.device)
        #         if i==0:
        #             gaussian = torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma[i]**2)*b[i]
        #         else:
        #             gaussian=gaussian+torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma[i]**2)*b[i]
        #     else:
        #         if i==0:
        #             gaussian = torch.exp(-0.5 * ((all_x[i]**2)/sigma[0][i]**2 + (all_y[i]**2)/sigma[1][i]**2) )*b[i]
        #         else:
        #             gaussian=gaussian+torch.exp(-0.5 * (all_x[i]**2 + all_y[i]**2) / sigma[i]**2)*b[i]

    if normalization:
        gaussian = gaussian / gaussian.max()
    
    # Plot the Gaussian mask
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z)
    # ax.view_init(elev=20., azim=35)
    # print(gaussian.shape)
    # print(gaussian)
    # print(sigma)
    # ax.plot_surface(grid_x, grid_y, gaussian, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.8)
    # plt.title('3D Gaussian Mask')
    # plt.xlabel('X axis')
    # plt.ylabel('Y axis')
    # ax.set_zlim(-1,1)  # Limit the Z axis for better visualization
    # plt.savefig("1.png")
    # plt.show()
    return gaussian

class MaskConv2d(nn.Module):
    def __init__(self, conv_type=1, sigma = 1.0, noise=True, normalization=True, offset=True, circle=True,*args, **kwargs):
        super().__init__()
        
        # assert conv_type in (1,2)
        # self.conv_type = conv_type
        # self.noise = noise 
        # self.normalization = normalization
        self.conv = nn.Conv2d(*args,**kwargs)
        _, _, KH, KW = self.conv.weight.data.size()
        # self.circle = circle
        # if offset:
        #     self.offset = nn.Parameter(torch.tensor([0.0,0.0],dtype=torch.float32),requires_grad=True)
        #     # self.offset = nn.Parameter(torch.zeros((2,KH*KW),dtype=torch.float32),requires_grad=True)
        # else:
        #     self.offset = []
        # # out_channels, in_channels, kH, kW = self.conv.weight.data.size()
        # if circle:
        #     # self.mask_sigma = nn.Parameter(torch.ones((self.conv.out_channels),dtype=torch.float32)*sigma)
        #     self.mask_sigma = nn.Parameter(torch.tensor(sigma,dtype=torch.float32),requires_grad=True)
        # else:
        #     # self.mask_sigma = nn.Parameter(torch.ones((2,self.conv.out_channels),dtype=torch.float32)*sigma)
        #     self.mask_sigma = nn.Parameter(torch.tensor([1.0,1.0],dtype=torch.float32)*sigma,requires_grad=True)
        # # self.mask = nn.Parameter(torch.ones((KH,KW),dtype=torch.float32).reshape(1,1,KH,KW),requires_grad=True)
        

        
        

        # 自由的mask
        self.mask = nn.Parameter((torch.ones((KH,KW),dtype=torch.float32).reshape(1,1,KH,KW)),requires_grad=True)
        # self.mask = nn.Parameter((torch.randn((KH,KW),dtype=torch.float32).reshape(1,1,KH,KW)),requires_grad=True)
        # self.mask_weight=[]
        # if conv_type==2:
        #     if KH==KW:
        #         self.mask_weight = nn.Parameter(torch.tensor([0.5/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=False)
        #         self.mask_weight[int((KH*KW)/2)] +=torch.tensor(0.5) 
        #     else:
        #         self.mask_weight = nn.Parameter(torch.tensor([1.0/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=True)
        #     self.mask_weight.requires_grad=True
        #     if circle==True:
        #         self.mask_sigma = nn.Parameter(torch.ones((KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
        #         # self.mask_sigma = nn.Parameter(torch.ones((KH*KW,self.conv.out_channels),dtype=torch.float32)*sigma,requires_grad=True)
        #     else:
        #         # self.mask_sigma = nn.Parameter(torch.ones((2,KH*KW,self.conv.out_channels),dtype=torch.float32)*sigma,requires_grad=True)
        #         self.mask_sigma = nn.Parameter(torch.ones((2,KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
        #     if offset:
        #         self.offset = nn.Parameter(torch.zeros((2,KH*KW),dtype=torch.float32),requires_grad=True)
        #         # self.offset = nn.Parameter(torch.zeros((2,KH*KW,KH*KW),dtype=torch.float32),requires_grad=True)
        #     else:
        #         self.offset = []
        # # print(self.mask_weight)
        # # print(self.mask_sigma)
        # # self.mask = create_gaussian_mask(kH,kW, level=0.1, sigma=sigma,mask_type=conv_type).reshape(1,1, kH, kW)
        # # print(mask)
        # # self.register_buffer('mask',mask,False)

    def forward(self,x):
        # print(self.conv.weight.data[0][0])
        # output_ch, _, kH, kW = self.conv.weight.data.size()
        # if type(kH) is not int:
        #     kH=kH.to(self.mask_sigma.device)
        #     kW=kW.to(self.mask_sigma.device)
        # # self.mask_sigma.data = torch.clamp(self.mask_sigma,min=1)
        # # if len(self.offset)>1:
        # #     self.offset.requires_grad=False
        # #     self.offset[0] = torch.clamp(self.offset[0],min=-0.5,max=0.5)
        # #     self.offset[1] = torch.clamp(self.offset[1],min=-0.5,max=0.5)
        # #     self.offset.requires_grad=True
        # mask = create_gaussian_mask(kH,kW, level=0.1, sigma=self.mask_sigma,mask_type=self.conv_type,weight= self.mask_weight,noise=self.noise,normalization=self.normalization,offset=self.offset,circle=self.circle).reshape(1,1, kH, kW)
        # masked_weights = self.conv.weight * mask
        # 
        masked_weights = self.conv.weight * self.mask
        # print(self.conv.weight.data[0][0])
        output = nn.functional.conv2d(x, masked_weights, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        # output = self.conv(x)
        return output



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

# def conv_bn(inp, oup, stride):
#     Conv2=MaskConv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1,  bias=False, conv_type=1,noise=False,normalization=True,offset=False,circle=True,sigma=5.0)
#     return nn.Sequential(
#         Conv2,
#         nn.BatchNorm2d(oup),
#         nn.ReLU(inplace=True)
#     )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class DynamicReLU(nn.Module):
    def __init__(self, total_epochs):
        super(DynamicReLU, self).__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0  # 初始epoch
    
    def forward(self, x):
        a = self.current_epoch / self.total_epochs  # 计算a值
        return torch.max(a * x, x)
    
    def update_epoch(self, epoch):
        self.current_epoch = min(epoch, self.total_epochs)  # 确保a不会超过1


class RepNeedleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,expand_rate = 2):
        super(RepNeedleBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.relu = DynamicReLU(240)
     
        # 不能引入bias，所以只能是identity
        self.skip = nn.Identity()
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                      padding=0, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
        else:
            self.needle = nn.Sequential(
                nn.Conv2d(in_channels,in_channels*expand_rate,1,1,0,bias=False,groups=groups),
                nn.BatchNorm2d(in_channels*expand_rate),
                self.relu,
                nn.Conv2d(in_channels*expand_rate,in_channels*expand_rate,1,1,0,bias=False,groups=groups),
                nn.BatchNorm2d(in_channels*expand_rate),
                self.relu,
                nn.Conv2d(in_channels*expand_rate,out_channels,1,1,0,bias=False,groups=groups),
                nn.BatchNorm2d(out_channels),
                self.relu
            )
            # self.rbr_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups,bias=False)


    def forward(self, inputs,epoch):
        if hasattr(self, 'rbr_reparam'):
            return self.rbr_reparam(inputs)
        self.needle.self.relu.update_epoch(epoch)
        return self.needle(inputs)+self.skip(inputs)
      

    #   Optional. This may improve the accuracy and facilitates quantization in some cases.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        # if stride == 2 :
        #     self.conv = nn.Sequential(
        #     # pw
        #     nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
        #     # RepNeedleBlock(inp*expand_ratio,inp*expand_ratio),
        #     # nn.Conv2d(inp * expand_ratio,inp*expand_ratio*2,1,1,0,bias=False),
        #     # nn.Conv2d(inp * expand_ratio*2,inp*expand_ratio*2,1,1,0,bias=False),
        #     # nn.Conv2d(inp * expand_ratio*2,inp * expand_ratio,1,1,0,bias=False),
        #     nn.BatchNorm2d(inp * expand_ratio),
        #     nn.ReLU(inplace=True),
        #     # dw
        #     nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
        #     RepNeedleBlock(inp*expand_ratio,inp*expand_ratio,groups=inp*expand_ratio,expand_rate=4),
        #     # nn.Conv2d(inp * expand_ratio,inp*expand_ratio*2,1,1,0,bias=False,groups=inp * expand_ratio),
        #     # nn.Conv2d(inp * expand_ratio*2,inp*expand_ratio*2,1,1,0,bias=False,groups=inp * expand_ratio),
        #     # nn.Conv2d(inp * expand_ratio*2,inp * expand_ratio,1,1,0,bias=False,groups=inp * expand_ratio),
        #     # DOConv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, groups=inp * expand_ratio,padding=1,bias=False),
        #     nn.BatchNorm2d(inp * expand_ratio),
        #     nn.ReLU(inplace=True),
        #     # pw-linear
        #     nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
        #     # RepNeedleBlock(oup,oup),
        #     # nn.Conv2d(oup,inp*expand_ratio*2,1,1,0,bias=False),
        #     # nn.Conv2d(inp * expand_ratio*2,inp*expand_ratio*2,1,1,0,bias=False),
        #     # nn.Conv2d(inp * expand_ratio*2,oup,1,1,0,bias=False),
        #     nn.BatchNorm2d(oup),
        # )
        # else:
        #     self.conv = nn.Sequential(
        #     # pw
        #     nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
        #     # RepNeedleBlock(inp*expand_ratio,inp*expand_ratio),
        #     # nn.Conv2d(inp * expand_ratio,inp*expand_ratio*2,1,1,0,bias=False),
        #     # nn.Conv2d(inp * expand_ratio*2,inp*expand_ratio*2,1,1,0,bias=False),
        #     # nn.Conv2d(inp * expand_ratio*2,inp * expand_ratio,1,1,0,bias=False),
        #     nn.BatchNorm2d(inp * expand_ratio),
        #     nn.ReLU(inplace=True),
        #     # dw
        #     nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
        #     RepNeedleBlock(inp*expand_ratio,inp*expand_ratio,groups=inp*expand_ratio,expand_rate=4),
        #     # nn.Conv2d(inp * expand_ratio,inp*expand_ratio*2,1,1,0,bias=False,groups=inp * expand_ratio),
        #     # nn.Conv2d(inp * expand_ratio*2,inp*expand_ratio*2,1,1,0,bias=False,groups=inp * expand_ratio),
        #     # nn.Conv2d(inp * expand_ratio*2,inp * expand_ratio,1,1,0,bias=False,groups=inp * expand_ratio),
        #     # DOConv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, groups=inp * expand_ratio,padding=1,bias=False),
        #     nn.BatchNorm2d(inp * expand_ratio),
        #     nn.ReLU(inplace=True),
        #     # pw-linear
        #     nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
        #     # RepNeedleBlock(oup,oup),
        #     # nn.Conv2d(oup,inp*expand_ratio*2,1,1,0,bias=False),
        #     # nn.Conv2d(inp * expand_ratio*2,inp*expand_ratio*2,1,1,0,bias=False),
        #     # nn.Conv2d(inp * expand_ratio*2,oup,1,1,0,bias=False),
        #     nn.BatchNorm2d(oup),
        # )

        # self.conv = nn.Sequential(
        #     # pw
        #     nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(inp * expand_ratio),
        #     nn.ReLU(inplace=True),
        #     # dw
        #     nn.Conv2d(inp, inp , 3, stride, 1, groups=1, bias=False),
        #     nn.BatchNorm2d(inp),
        #     nn.ReLU(inplace=True),
        #     # pw-linear
        #     nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(oup),
        # )
        # Conv2 = MaskConv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio, kernel_size=3, stride=stride, padding=1, groups=inp * expand_ratio, bias=False, conv_type=1,noise=False,normalization=True,offset=False,circle=True,sigma=5.0)
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            # Conv2,
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x
        if self.use_res_connect:
            return t + self.conv(x)
        else:
            return self.conv(x)


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
        # self.interverted_residual_setting = [
        #     # t, c, n, s
        #     [1, 16, 1, 1],
        #     [T, 24, 2, 1],
        #     [T, 32, 3, 2],
        #     [T, 64, 4, 1],
        #     [T, 96, 3, 1],
        #     [T, 160, 3, 2],
        #     [T, 320, 1, 1],
        # ]
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 2],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]


        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel, 2)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else int(c*width_mult)*4
        self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, feature_dim),
        )

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
        f00=x
        out = self.conv1(x)
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

        out = self.conv2(out)

        if not self.remove_avg:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.classifier(out)

        if is_feat:
            return [f00,f0, f1, f2, f3, f4, f5], out
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


def mobile_half(num_classes):
    return mobilenetv2_T_w(6, 0.5, num_classes)
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import tracemalloc
if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)

    net = mobile_half(100)
    tracemalloc.start()
    with torch.no_grad():
        feats, logit = net(x, is_feat=True, preact=True)
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    # 找到峰值内存占用
    peak_memory = max(stat.size / (1024 * 1024) for stat in top_stats)
    print(f"Peak memory usage: {peak_memory:.2f} MB")

    # 停止内存分配跟踪
    tracemalloc.stop()
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
    flops = FlopCountAnalysis(net,x)
    params = parameter_count_table(net)
    print(flops.total())
    print(params)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')


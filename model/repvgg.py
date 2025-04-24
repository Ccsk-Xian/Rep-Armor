# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch.nn as nn
import numpy as np
import torch
import copy
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


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
            
        #     if len(offset)>1:
        #         offset = offset.view(2,KH*KW,1,1)
        #         # gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2 + (grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2) / sigma[i*KH+o]**2)*b[i*KH+o]
        #         gaussian = torch.exp(-0.5 * ((grid_x-i-offset[0])**2 + (grid_y-o-offset[1])**2) / sigma**2)
        #     else:
        #         # print(grid_x.shape)
        #         # print(sigma[i*KH+o].shape)
        #         gaussian = torch.exp(-0.5 * ((grid_x-i)**2 + (grid_y-o)**2) / sigma**2)
        # else:
        #     sigma = sigma.view(2,KH*KW,1,1)
        #     if len(offset)>1:
        #         offset = offset.view(2,KH*KW,1,1)
        #         # gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0][i*KH+o].reshape(grid_x.shape))**2)/sigma[0][i*KH+o]**2 + ((grid_y-o-offset[1][i*KH+o].reshape(grid_x.shape))**2)/sigma[1][i*KH+o]**2))*b[i*KH+o]
        #         gaussian = torch.exp(-0.5 * (((grid_x-i-offset[0])**2)/sigma[0]**2 + ((grid_y-o-offset[1])**2)/sigma[1]**2))
        #     else:
        #         gaussian = torch.exp(-0.5 * (((grid_x-i)**2)/sigma[0]**2 + ((grid_y-o)**2)/sigma[1]**2))

        # gaussian = gaussian.sum(dim=0)
        # gaussian = gaussian/(KH*KW)
        # print(gaussian.shape)

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
        
        assert conv_type in (1,2)
        self.conv_type = conv_type
        self.noise = noise 
        self.normalization = normalization
        self.conv = nn.Conv2d(*args,**kwargs)
        _, _, KH, KW = self.conv.weight.data.size()
        self.circle = circle
        if offset:
            self.offset = nn.Parameter(torch.tensor([0.0,0.0],dtype=torch.float32),requires_grad=True)
            # self.offset = nn.Parameter(torch.zeros((2,KH*KW),dtype=torch.float32),requires_grad=True)
        else:
            self.offset = []
        # out_channels, in_channels, kH, kW = self.conv.weight.data.size()
        if circle:
            # self.mask_sigma = nn.Parameter(torch.ones((self.conv.out_channels),dtype=torch.float32)*sigma)
            self.mask_sigma = nn.Parameter(torch.tensor(sigma,dtype=torch.float32),requires_grad=True)
        else:
            # self.mask_sigma = nn.Parameter(torch.ones((2,self.conv.out_channels),dtype=torch.float32)*sigma)
            self.mask_sigma = nn.Parameter(torch.tensor([1.0,1.0],dtype=torch.float32)*sigma,requires_grad=True)
        # self.mask = nn.Parameter(torch.ones((KH,KW),dtype=torch.float32).reshape(1,1,KH,KW),requires_grad=True)
        

        
        

        # 自由的mask
        # self.mask = nn.Parameter((torch.ones((KH,KW),dtype=torch.float32).reshape(1,1,KH,KW)),requires_grad=True)
        # self.mask = nn.Parameter((torch.randn((KH,KW),dtype=torch.float32).reshape(1,1,KH,KW)),requires_grad=True)
        self.mask_weight=[]
        if conv_type==2:
            if KH==KW:
                self.mask_weight = nn.Parameter(torch.tensor([1.0/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=True)
                # self.mask_weight[int((KH*KW)/2)] +=torch.tensor(0.5) 
            else:
                self.mask_weight = nn.Parameter(torch.tensor([1.0/(KH*KW) for i in range((KH*KW))],dtype=torch.float32),requires_grad=True)
            # self.mask_weight.requires_grad=True
            if circle==True:
                self.mask_sigma = nn.Parameter(torch.ones((KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
                # self.mask_sigma = nn.Parameter(torch.ones((KH*KW,self.conv.out_channels),dtype=torch.float32)*sigma,requires_grad=True)
            else:
                # self.mask_sigma = nn.Parameter(torch.ones((2,KH*KW,self.conv.out_channels),dtype=torch.float32)*sigma,requires_grad=True)
                self.mask_sigma = nn.Parameter(torch.ones((2,KH*KW),dtype=torch.float32)*sigma,requires_grad=True)
            if offset:
                self.offset = nn.Parameter(torch.zeros((2,KH*KW),dtype=torch.float32),requires_grad=True)
                # self.offset = nn.Parameter(torch.zeros((2,KH*KW,KH*KW),dtype=torch.float32),requires_grad=True)
            else:
                self.offset = []
        # print(self.mask_weight)
        # print(self.mask_sigma)
        # self.mask = create_gaussian_mask(kH,kW, level=0.1, sigma=sigma,mask_type=conv_type).reshape(1,1, kH, kW)
        # print(mask)
        # self.register_buffer('mask',mask,False)

    def forward(self,x):
        # print(self.conv.weight.data[0][0])
        output_ch, _, kH, kW = self.conv.weight.data.size()
        if type(kH) is not int:
            kH=kH.to(self.mask_sigma.device)
            kW=kW.to(self.mask_sigma.device)
        # self.mask_sigma.data = torch.clamp(self.mask_sigma,min=1)
        # if len(self.offset)>1:
        #     self.offset.requires_grad=False
        #     self.offset[0] = torch.clamp(self.offset[0],min=-0.5,max=0.5)
        #     self.offset[1] = torch.clamp(self.offset[1],min=-0.5,max=0.5)
        #     self.offset.requires_grad=True
        mask = create_gaussian_mask(kH,kW, level=0.1, sigma=self.mask_sigma,mask_type=self.conv_type,weight= self.mask_weight,noise=self.noise,normalization=self.normalization,offset=self.offset,circle=self.circle).reshape(1,1, kH, kW)
        masked_weights = self.conv.weight * mask
        # 
        # masked_weights = self.conv.weight * self.mask
        # print(self.conv.weight.data[0][0])
        output = nn.functional.conv2d(x, masked_weights, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        # output = self.conv(x)
        return output
    # def post_op_mask(self,optimizer):

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
    
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    # result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
    #                                               kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    # MaskConv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1,  bias=False, conv_type=mask_type,noise=False,normalization=True,offset=True,circle=False,sigma=5.0)
    if kernel_size==1:
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    elif kernel_size>=3:   
        result.add_module('conv',MaskConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,groups=groups,bias=False,
                                            conv_type=2,noise=False,normalization=True,offset=False,circle=True,sigma=5.0))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            
            # self.rbr_identity = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense2 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            # self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            print('RepVGG Block, identity = ', self.rbr_identity)
        # self.bnend = nn.BatchNorm2d(out_channels)
        # self.div = torch.tensor([3.0]) if out_channels == in_channels and stride == 1 else torch.tensor([2.0])
        self.register_buffer('div', torch.tensor([3.0 if out_channels == in_channels and stride == 1 else 2.0]))
        
    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        # return self.nonlinearity(self.se((self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)/self.div))
        return self.nonlinearity(self.se((self.rbr_dense(inputs) +self.rbr_dense2(inputs)+id_out)/self.div))
        # return self.bnend(self.nonlinearity(self.se((self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))))


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



class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False, use_checkpoint=False):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
            
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_A0(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=100,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_A1(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_A2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_B0(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_B1(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_B1g2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_B1g4(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_B2g2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_B2g4(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B3(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_B3g2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_B3g4(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint)

def create_RepVGG_D2se(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True, use_checkpoint=use_checkpoint)


func_dict = {
'RepVGG-A0': create_RepVGG_A0,
'RepVGG-A1': create_RepVGG_A1,
'RepVGG-A2': create_RepVGG_A2,
'RepVGG-B0': create_RepVGG_B0,
'RepVGG-B1': create_RepVGG_B1,
'RepVGG-B1g2': create_RepVGG_B1g2,
'RepVGG-B1g4': create_RepVGG_B1g4,
'RepVGG-B2': create_RepVGG_B2,
'RepVGG-B2g2': create_RepVGG_B2g2,
'RepVGG-B2g4': create_RepVGG_B2g4,
'RepVGG-B3': create_RepVGG_B3,
'RepVGG-B3g2': create_RepVGG_B3g2,
'RepVGG-B3g4': create_RepVGG_B3g4,
'RepVGG-D2se': create_RepVGG_D2se,      #   Updated at April 25, 2021. This is not reported in the CVPR paper.
}




#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True

#   ====================== for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
#   =====================   example_pspnet.py shows an example

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

def repvggnet(name='RepVGG-A0',num_classes=100):
    net = func_dict[name]
    net = net()
    return net

from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile
if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)

    net = repvggnet('RepVGG-A0')
    # net = net()

    logit = net(x)
    # for f in feats:
    #     print(f.shape, f.min().item())
    print(logit.shape)
    flops = FlopCountAnalysis(net,x)
    params = parameter_count_table(net)
    print(flops.total())
    print(params)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    # for m in net.get_bn_before_relu():
    #     if isinstance(m, nn.BatchNorm2d):
    #         print('pass')
    #     else:
    #         print('warning')

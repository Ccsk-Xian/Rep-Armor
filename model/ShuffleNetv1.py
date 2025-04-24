'''ShuffleNet in PyTorch.
See the paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from timm.models.vision_transformer import trunc_normal_
from .convnet_utils import conv_bn_relu

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
                    nn.Conv2d(int(in_channels*expand_rate),int(in_channels*expand_rate),1,1,0,bias=False,groups=groups),
                    nn.BatchNorm2d(int(in_channels*expand_rate)),
                    nn.Conv2d(int(in_channels*expand_rate),out_channels,1,1,0,bias=False,groups=groups),
                    # nn.BatchNorm2d(out_channels),
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
                    nn.Conv2d(in_channels,in_channels,1,1,0,bias=False,groups=groups),
                    nn.BatchNorm2d(out_channels),
                )
            
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

def conv_bn_rep(in_channels, out_channels, kernel_size, stride, padding, groups=1,use_bn=True,follow=False,pattern=3):
    result = nn.Sequential()
    # if follow:
    #     result.add_module('conv1',RepTowerBlock(in_channels=in_channels,out_channels=in_channels,groups=groups,pattern=pattern,expand_rate=2))
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
import math

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,use_act=True,use_bn=True,use_identity=True):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.use_act = use_act
        self.count = Parameter(torch.Tensor(1),requires_grad=True)
        torch.nn.init.constant_(self.count,1.0)
       
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
            self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False,pattern=3)
            # self.dense_weight_2 = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
            self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False,pattern=1)
            self.rbr_dense_3 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False,pattern=2)
            self.rbr_dense_4 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False,pattern=1)
            # self.rbr_dense_5 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False,pattern=1)
            # self.tower = RepTowerBlock(out_channels,out_channels,1,0,1,groups)
            # self.rbr_dense_5 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False,pattern=3)
            # self.rbr_dense_6 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False,pattern=3)
            
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
        # return self.nonlinearity(self.se(self.rbr_dense(inputs)+self.rbr_dense_2(inputs)+self.rbr_dense_3(inputs)+self.rbr_dense_4(inputs)+self.rbr_dense_5(inputs))/5)
        return self.nonlinearity(self.se(self.rbr_dense(inputs)+self.rbr_dense_2(inputs)+self.rbr_dense_3(inputs)+self.rbr_dense_4(inputs)))
        # return self.nonlinearity(self.se(self.rbr_dense(inputs)+self.rbr_dense_2(inputs)+self.rbr_dense_3(inputs)+self.rbr_dense_4(inputs)+self.rbr_dense_5(inputs)))
        # return self.nonlinearity(self.se(self.dense_weight*self.rbr_dense(inputs) + self.rbr_1x1_weight*self.rbr_1x1(inputs)+id_out))
# def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
#     result = nn.Sequential()
#     result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                                   kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
#     result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
#     return result

# class RepVGGBlock1(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
#         super(RepVGGBlock1, self).__init__()
#         self.deploy = deploy
#         self.groups = groups
#         self.in_channels = in_channels

#         # assert kernel_size == 3
#         # assert padding == 1

#         padding_11 = padding - kernel_size // 2

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
#             self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
#             self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
#             print('RepVGG Block, identity = ', self.rbr_identity)


#     def forward(self, inputs):
#         if hasattr(self, 'rbr_reparam'):
#             return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

#         if self.rbr_identity is None:
#             id_out = 0
#         else:
#             id_out = self.rbr_identity(inputs)

#         return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

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
                                RepTowerBlock(outchannels,outchannels,groups=group,expand_rate=4))
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
            out = out +inner_conv1(bn(conv(x)))
            # out = out+bn(conv(x))
        # return out/len(self.kernel_sizes)
        return out/math.sqrt(len(self.kernel_sizes))
        # return out 

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        # self.add_module('bn', torch.nn.BatchNorm1d(a))
        # self.add_module('dropout',torch.nn.Dropout(p=0.2))
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
            # self.classifier_dist_2 = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
            
            
    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            
            x = (x[0] + x[1])/2 
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

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.stride = stride

        mid_planes = int(out_planes/4)
        g = 1 if in_planes == 24 else groups
        # self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        # self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv1 = RepVGGBlock(in_channels=in_planes, out_channels=mid_planes, groups=g, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False,use_bn=True)
        # self.conv1 = RepVGGBlock1(in_channels=in_planes, out_channels=mid_planes,groups=g, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False)
        # self.conv1 = conv_bn_relu(in_planes,mid_planes,1,1,0,1,g)
        self.shuffle1 = ShuffleBlock(groups=g)
        # self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        # self.conv2 = conv_bn_relu(mid_planes,mid_planes,3,stride,1,1,mid_planes)
        self.conv2 = RepVGGBlock(in_channels=mid_planes, out_channels=mid_planes, groups=mid_planes, kernel_size=3, stride=stride, padding=1, deploy=False, use_se=False,use_bn=True)
        # self.conv2 = RepVGGBlock1(in_channels=mid_planes, out_channels=mid_planes,groups=mid_planes, kernel_size=3, stride=stride, padding=1, deploy=False, use_se=False)
        # self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_planes)
        # self.conv3 = RepVGGBlock(in_channels=mid_planes, out_channels=out_planes, groups=groups, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False,use_bn=False)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv1(x)
        out = self.shuffle1(out)
        # out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv2(out)
        # out = self.bn3(self.conv3(out))
        out = self.conv3(out)
        # out = self.conv1(x)
        # out = self.shuffle1(out)
        # out = self.conv2(out)
        # out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        preact = torch.cat([out, res], 1) if self.stride == 2 else out+res
        out = F.relu(preact)
        # out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out+res)
        if self.is_last:
            return out, preact
        else:
            return out


class ShuffleNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        # self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)

        # self.conv1 = nn.Conv2d(
        # 3, 
        # 24, 
        # kernel_size=3, 
        # stride=2,
        # padding=1,
        # bias=False,
        # )
        self.conv1 =  RepVGGBlock(in_channels=3, out_channels=24, groups=1, kernel_size=3, stride=1, padding=1, deploy=False, use_se=False,use_bn=True)
        # self.conv1 = RepVGGBlock1(in_channels=3, out_channels=24,groups=1, kernel_size=3, stride=1, padding=1, deploy=False, use_se=False)
        # self.conv1 = Rep_bn(inchannels=3,outchannels=3,kernel_size=7,stride=1,bias=False,group=1)
        # self.conv1 = conv_bn_relu(3,24,3,1,1)
        # self.conv1_follow = conv_bn_relu(3,24,3,1,1)
        # self.conv1_follow = nn.Conv2d(
        # 3, 
        # 24, 
        # kernel_size=3, 
        # stride=2,
        # padding=1,
        # bias=False,
        # )
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], num_classes)
        # self.linear = Classfier(out_planes[2], num_classes,True)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes,
                                     stride=stride,
                                     groups=groups,
                                     is_last=(i == num_blocks - 1)))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        raise NotImplementedError('ShuffleNet currently is not supported for "Overhaul" teacher')

    def forward(self, x, is_feat=False, preact=False):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.conv1_follow(F.relu(self.conv1(x)))
        # out = self.conv1_follow(x)
        out = self.conv1(x)
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.maxpool(out)
        f0 = out
        out, f1_pre = self.layer1(out)
        f1 = out
        out, f2_pre = self.layer2(out)
        f2 = out
        out, f3_pre = self.layer3(out)
        f3 = out
        # out = F.avg_pool2d(out, 4)
        # print()
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        f4 = out
        out = self.linear(out)

        # out = F.relu(self.bn1(self.conv1(x)))
        # f0 = out
        # out, f1_pre = self.layer1(out)
        # f1 = out
        # out, f2_pre = self.layer2(out)
        # f2 = out
        # out, f3_pre = self.layer3(out)
        # f3 = out
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # f4 = out
        # out = self.linear(out)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4], out
            else:
                return [f0, f1, f2, f3, f4], out
        else:
            return out


def ShuffleV1(**kwargs):
    cfg = {
        'out_planes': [240, 480, 960],
        'num_blocks': [4, 8, 4],
        'groups': 3
    }
    return ShuffleNet(cfg, **kwargs)

from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)

    net = ShuffleV1(num_classes=100)

    feats, logit = net(x, is_feat=True, preact=True)
    for f in feats:
        print(f.shape, f.min().item())
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


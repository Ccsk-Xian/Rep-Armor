import math
import torch
import torch.nn as nn
# from utee import misc
from collections import OrderedDict
from .convnet_utils import conv_bn_relu
from torch.nn.parameter import Parameter
__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}
class RepTowerBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,expand_rate = 3,pattern = 3):
        super(RepTowerBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

     
        # 不能引入bias，所以只能是identity
        # self.skip = nn.Identity() if in_channels==out_channels else nn.BatchNorm2d(in_channels)
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
        # self.origin_1x1 = RepTowerBlock(outchannels,outchannels,groups=group,expand_rate=4)
        self.origin_1x1 = RepTowerBlock(inchannels,inchannels,groups=group,expand_rate=4)
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
                # self.__setattr__('conv_one{}_{}'.format(k, r),
                #                 RepTowerBlock(outchannels,outchannels,groups=group,expand_rate=4))
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
            out = out +inner_conv1(bn(conv(x)))
            # out = out +bn(conv(inner_conv1(x)))
            # out = out+bn(conv(x))
        # return out
        # return out/len(self.kernel_sizes)
        return out/math.sqrt(len(self.kernel_sizes))
        # return bn1(inner_conv1(out))


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
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock1, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        # assert kernel_size == 3
        # assert padding == 1

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
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

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
            self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=3)
            # self.dense_weight_2 = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
            self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=1)
            self.rbr_dense_3 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=2)
            self.rbr_dense_4 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False,pattern=1)
            self.rbr_dense_5 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False,pattern=1)
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
        return self.nonlinearity(self.se(self.rbr_dense(inputs)+self.rbr_dense_2(inputs)+self.rbr_dense_3(inputs)+self.rbr_dense_4(inputs)+self.rbr_dense_5(inputs))/self.count)
        # return self.nonlinearity(self.se(self.rbr_dense(inputs)+self.rbr_dense_2(inputs)+self.rbr_dense_3(inputs)+self.rbr_dense_4(inputs))/self.count)
        # return self.nonlinearity(self.se(self.rbr_dense(inputs)+self.rbr_dense_2(inputs)+self.rbr_dense_3(inputs)+self.rbr_dense_4(inputs)+self.rbr_dense_5(inputs)+self.rbr_dense_6(inputs))/6)
        # return self.nonlinearity(self.se(self.dense_weight*self.rbr_dense(inputs) + self.rbr_1x1_weight*self.rbr_1x1(inputs)+id_out))

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
        self.add_module('drop',torch.nn.Dropout(p=0.5))
        # self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        self.add_module("l",torch.nn.Conv2d(a,b,1,bias=bias))
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

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes

        self.group1 = nn.Sequential(
            OrderedDict([
                # ('squeeze', nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)),
                # ('squeeze_activation', nn.ReLU(inplace=True))
                # ('squeeze',conv_bn_relu(inplanes,squeeze_planes,1,1,0)),
                ('squeeze', RepVGGBlock(in_channels=inplanes, out_channels=squeeze_planes, groups=1, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False,use_bn=True)),
                # ('squeeze', RepVGGBlock1(in_channels=inplanes, out_channels=squeeze_planes,groups=1, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False)),
            ])
        )

        self.group2 = nn.Sequential(
            OrderedDict([
                # ('expand1x1', nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)),
                # ('expand1x1_activation', nn.ReLU(inplace=True))
                # ('squeeze',conv_bn_relu(squeeze_planes,expand1x1_planes,1,1,0)),
                ('expand1x1_activation', RepVGGBlock(in_channels=squeeze_planes, out_channels=expand1x1_planes, groups=1, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False,use_bn=True))
                # ('expand1x1_activation', RepVGGBlock1(in_channels=squeeze_planes, out_channels=expand1x1_planes,groups=1, kernel_size=1, stride=1, padding=0, deploy=False, use_se=False)),
            ])

        )

        self.group3 = nn.Sequential(
            OrderedDict([
                # ('expand3x3', nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)),
                ('expand3x3',conv_bn_relu(squeeze_planes,expand3x3_planes,3,1,1)),
                # ('expand3x3',RepVGGBlock(in_channels=squeeze_planes, out_channels=expand3x3_planes, groups=1, kernel_size=3, stride=1, padding=1, deploy=False, use_se=False,use_bn=True)),
                # ('expand3x3', RepVGGBlock1(in_channels=squeeze_planes, out_channels=expand3x3_planes,groups=1, kernel_size=3, stride=1, padding=1, deploy=False, use_se=False)),
                # ('expand3x3_activation', nn.ReLU(inplace=True))
            ])
        )

    def forward(self, x):
        x = self.group1(x)
        return torch.cat([self.group2(x),self.group3(x)], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                # Rep_bn(inchannels=3,outchannels=3,kernel_size=7,stride=1,bias=False,group=1),
                # nn.ReLU(inplace=True),
                # conv1_bn(3,96,1),
                nn.Conv2d(3, 96, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                Rep_bn(inchannels=3,outchannels=3,kernel_size=7,stride=1,bias=False,group=1),
                nn.ReLU(inplace=True),
                conv_bn_relu(3,64,3,2,1),
                # conv_bn_relu(3,64,3,2,1),
                # RepVGGBlock(in_channels=3, out_channels=64, groups=1, kernel_size=3, stride=1, padding=1, deploy=False, use_se=False,use_bn=True),
                # RepVGGBlock1(in_channels=3, out_channels=64,groups=1, kernel_size=3, stride=1, padding=1, deploy=False, use_se=False),
                # conv1_bn(3,64,1),
                # nn.Conv2d(3, 64, kernel_size=3, stride=1),
                # nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        # final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        final_conv = Classfier(512, num_classes, True)
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.2),
            final_conv,
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(13)
            nn.AdaptiveAvgPool2d((1, 1))

        )
        # self.classifier = Classfier(512, num_classes, True)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = 2.0
                # if m is final_conv:
                #     m.weight.data.normal_(0, 0.01)
                # else:
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                u = math.sqrt(3.0 * gain / fan_in)
                m.weight.data.uniform_(-u, u)
                if m.bias is not None:
                    m.bias.data.zero_()

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     # return x
    #     return x.view(x.size(0), self.num_classes)
    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)
    
def squeezenet1_0(pretrained=False, model_root=None, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    model = SqueezeNet(version=1.0, **kwargs)
    # if pretrained:
    #     misc.load_state_dict(model, model_urls['squeezenet1_0'], model_root)
    return model


def squeezenet1_1(pretrained=False, model_root=None, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    model = SqueezeNet(version=1.1, **kwargs)
    # if pretrained:
    #     misc.load_state_dict(model, model_urls['squeezenet1_1'], model_root)
    return model

def squeezenet1(num_classes):
    return squeezenet1_1(num_classes=num_classes)

from fvcore.nn import FlopCountAnalysis, parameter_count_table
if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)

    net = squeezenet1_1(100)

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
"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>
"""

import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
__all__ = ['mobilenetv2_T_w', 'mobile_half']

BN = None
def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)
    
class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        # 这个本质上其实还是并行的，**看上去是使用了更少的参数来代替大的卷积核，dilated只是增加多样性的一种方式。还是没有用weights这里可以试一下，穷举orweight
        # if kernel_size == 17:
        #     self.kernel_sizes = [5, 9, 3, 3, 3]
        #     self.dilates = [1, 2, 4, 5, 7]
        # elif kernel_size == 15:
        #     self.kernel_sizes = [5, 7, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 5, 7]
        # elif kernel_size == 13:
        #     self.kernel_sizes = [5, 7, 3, 3, 3,1]
        #     self.dilates = [1, 2, 3, 4, 5,1]
        # elif kernel_size == 11:
        #     self.kernel_sizes = [5, 5, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 4, 5]
        # elif kernel_size == 9:
        #     self.kernel_sizes = [5, 5, 3, 3]
        #     self.dilates = [1, 2, 3, 4]
        # elif kernel_size == 7:
        #     self.kernel_sizes = [5, 3, 3]
        #     self.dilates = [1, 2, 3]
        # elif kernel_size == 5:
        #     self.kernel_sizes = [3, 3]
        #     self.dilates = [1, 2]
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3,1]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3,1]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [11,9,5, 7, 3]
            self.dilates = [1, 2, 3, 4, 5,6]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 3,1]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 3,1]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3,1]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3,1]
            self.dilates = [1, 2]
        
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        self.count = 0
        
        # self.switch = Parameter(torch.Tensor(switch_num),requires_grad=True)
        self.k = []
        self.d = []
        self.origin_kernel_sizes=kernel_size
        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k in self.kernel_sizes:
                for r in self.dilates:
                    if 1+(k-1)*r<=self.origin_kernel_sizes:
                        if k ==1 and 1 in self.k:
                            continue
                        self.count+=1
                        self.k.append(k)
                        self.d.append(r)
        for k, r in zip(self.k, self.d):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))
                self.__setattr__('dil_weight_k{}_{}'.format(k, r), Parameter(torch.Tensor(1,channels,1,1),requires_grad=True))
                torch.nn.init.constant_(self.__getattr__('dil_weight_k{}_{}'.format(k, r)),1.0)
                
        self.origin_weight = Parameter(torch.Tensor(1,channels,1,1),requires_grad=True)
        self.bias = Parameter(torch.Tensor(1,channels,1,1),requires_grad=True)
        torch.nn.init.constant_(self.origin_weight,1.0)
        torch.nn.init.constant_(self.bias,0.0)
        

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))*self.origin_weight
        for i, (k, r) in enumerate(zip(self.k, self.d)):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            weight = self.__getattr__('dil_weight_k{}_{}'.format(k, r))
            # 可以试试动态减去某个分支，当前直接相乘就行了。
            out = out + bn(conv(x))*weight
        return out+self.bias
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


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,kernel_size):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size, stride, kernel_size//2, groups=inp * expand_ratio, bias=False),
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
        self.interverted_residual_setting = [
            # t, c, n, s,num
            [1, 16, 1, 1,3],
            [T, 24, 2, 2,3],
            [T, 32, 3, 2,13],
            [T, 64, 4, 2,13],
            [T, 96, 3, 1,3],
            [T, 160, 3, 2,7],
            [T, 320, 1, 1,3],
        ]


        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel, 2)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s, num in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t,num)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
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


def mobilecontainer(num_classes):
    return mobilenetv2_T_w(6, 0.5, num_classes)


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)

    net = mobilecontainer(100)

    feats, logit = net(x, is_feat=True, preact=True)
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')


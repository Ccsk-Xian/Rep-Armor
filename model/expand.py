"""
MobileNetV2 and its ExpandNet-CL with er (expansion_rate) = 4
"""
__all__ = ['MobileNetV2', 'mobilenetv2',
           'MobileNetV2_Expand', 'mobilenetv2_expand'
           ]

import torch.nn as nn
import math
import torch
import torch.nn.functional as F
# expansion rate
er = int(4)  # for all expand units.

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
        # return self.needle(inputs)+self.skip(inputs)
        return self.needle(inputs)

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_bn_expand(inp, oup, stride, er=er):
    
    # return nn.Sequential(
    #     #nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
    #     nn.Conv2d(inp, inp*er, 1, stride=1, padding=1, bias=False),
    #     nn.Conv2d(inp*er, oup*er, 3, stride=stride, padding=0, bias=False),
    #     nn.Conv2d(oup*er, oup, 1, stride=1, padding=0, bias=False),
    #     nn.BatchNorm2d(oup),
    #     nn.ReLU6(inplace=True)
    # )
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual_Expand(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_Expand, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                
                # nn.Conv2d(hidden_dim, hidden_dim*er, kernel_size=1, stride=1, padding=1, groups=hidden_dim, bias=False),
                # nn.Conv2d(hidden_dim*er, hidden_dim*er, kernel_size=3, stride=stride, padding=0, groups=hidden_dim, bias=False),
                # nn.Conv2d(hidden_dim*er, hidden_dim, kernel_size=1, stride=1, padding=0, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            self.first = RepTowerBlock(hidden_dim,hidden_dim,groups=hidden_dim,expand_rate=4)
            # self.second = RepTowerBlock(hidden_dim,hidden_dim,groups=hidden_dim,expand_rate=4,pattern=2)
            # self.third = RepTowerBlock(hidden_dim,hidden_dim,groups=hidden_dim,expand_rate=4,pattern=1)
            self.re = nn.ReLU6(inplace=True)
                # pw-linear
            self.pw=nn.Sequential(
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.ReLU6(inplace=True),
                # dw
                # nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # nn.Conv2d(hidden_dim, hidden_dim * er, kernel_size=1, stride=1, padding=0, groups=hidden_dim, bias=False),
                # # nn.BatchNorm2d(hidden_dim * er),
                # BNAndPadLayer(pad_pixels=3//2, num_features=hidden_dim*er),
                # nn.Conv2d(hidden_dim * er, hidden_dim * er, kernel_size=3, stride=stride, padding=0, groups=hidden_dim, bias=False),
                # nn.BatchNorm2d(hidden_dim * er),
                # # BNAndPadLayer(pad_pixels=3//2, num_features=hidden_dim*er),
                # nn.Conv2d(hidden_dim * er, hidden_dim, kernel_size=1, stride=1, padding=0, groups=hidden_dim, bias=False),
                # nn.BatchNorm2d(hidden_dim),
               )
            self.first = RepTowerBlock(hidden_dim,hidden_dim,groups=1,expand_rate=4)
            # self.second = RepTowerBlock(hidden_dim,hidden_dim,groups=1,expand_rate=2,pattern=2)
            # self.third = RepTowerBlock(hidden_dim,hidden_dim,groups=1,expand_rate=2,pattern=1)
            self.re = nn.ReLU6(inplace=True)
            self.pw = nn.Sequential(
                # pw-linear
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            out = self.conv(x)
            # return x + self.pw(self.re((self.first(out)+self.second(out)+self.third(out))/3.))
            return x + self.pw(self.re(self.first(out)))
        else:
            out = self.conv(x)
            # return self.pw(self.re((self.first(self.first(out)+self.second(out)+self.third(out))/3.)))
            return self.pw(self.re(self.first(out)))


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        self.features = nn.Sequential(*self.features)
        self.first = RepTowerBlock(input_channel,input_channel,groups=1,expand_rate=4)
        # self.second = RepTowerBlock(input_channel,input_channel,groups=1,expand_rate=4,pattern=2)
        # self.third = RepTowerBlock(input_channel,input_channel,groups=1,expand_rate=4,pattern=1)
        self.features1 = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features1.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features1.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features1.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features1 = nn.Sequential(*self.features1)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.first(x)+self.second(x)+self.third(x)
        x = self.features1(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

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


class MobileNetV2_Expand(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2_Expand, self).__init__()
        expand_block = InvertedResidual_Expand
        block = InvertedResidual

        input_channel = 32
        # last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        # self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn_expand(3, input_channel, 2)]
        self.features = nn.Sequential(*self.features)
        self.first = RepTowerBlock(input_channel,input_channel,groups=1,expand_rate=4)
        # self.second = RepTowerBlock(input_channel,input_channel,groups=1,expand_rate=4,pattern=2)
        # self.third = RepTowerBlock(input_channel,input_channel,groups=1,expand_rate=4,pattern=1)
        self.re = nn.ReLU6(inplace=True)
        self.features1 = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    if s == 2:
                        self.features1.append(expand_block(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features1.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features1.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
            # output_channel = int(c * width_mult)
            # for i in range(n):
            #     if i == 0:
            #         if s == 2:
            #             self.features1.append(expand_block(input_channel, output_channel, s, expand_ratio=t))
            #         else:
            #             self.features1.append(block(input_channel, output_channel, s, expand_ratio=t))
            #     else:
            #         self.features1.append(block(input_channel, output_channel, 1, expand_ratio=t))
            #     input_channel = output_channel
        # building last several layers
        # self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features1 = nn.Sequential(*self.features1)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.re((self.first(x)+self.second(x)+self.third(x))/3.)
        x = self.re(self.first(x))
        x = self.features1(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

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


def mobilenetv2(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        raise Exception('No pretrained url for expandnet')
    return model


def mobilenetv2_expand(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2_Expand(n_class=100, input_size=224, width_mult=0.66)
    if pretrained:
        raise Exception('No pretrained url for expandnet')
    return model
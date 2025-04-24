"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""

from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from .convnet_utils import conv_bn_relu
def get_config(mode: str = "xxs") -> dict:
    if mode == "tiny":
        mv2_exp_mult = 2
        config = {
            "layer1": {
                "out_channels": 16,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 24,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 48,
                "transformer_channels": 64,
                "ffn_dim": 128,
                "transformer_blocks": 1,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 64,
                "transformer_channels": 80,
                "ffn_dim": 160,
                "transformer_blocks": 2,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            # "layer5": {
            #     "out_channels": 80,
            #     "expand_ratio": mv2_exp_mult,
            #     "num_blocks": 3,
            #     "stride": 2,
            #     "last":True,
            #     "block_type": "mv2",
            # },
            "layer5": {  # 7x7
                "out_channels": 80,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 1,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    elif mode == "xx_small":
        mv2_exp_mult = 2
        config = {
            "layer1": {
                "out_channels": 16,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 24,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 48,
                "transformer_channels": 64,
                "ffn_dim": 128,
                "transformer_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 64,
                "transformer_channels": 80,
                "ffn_dim": 160,
                "transformer_blocks": 4,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 80,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    elif mode == "x_small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 48,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 64,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 80,
                "transformer_channels": 120,
                "ffn_dim": 240,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    elif mode == "small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 128,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 160,
                "transformer_channels": 240,
                "ffn_dim": 480,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    else:
        raise NotImplementedError

    for k in ["layer1", "layer2", "layer3", "layer4", "layer5"]:
        config[k].update({"dropout": 0.1, "ffn_dropout": 0.0, "attn_dropout": 0.0})

    return config


class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (float): Attention dropout. Default: 0.0
        bias (bool): Use bias or not. Default: ``True``

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        bias: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x_q: Tensor) -> Tensor:
        # [N, P, C]
        b_sz, n_patches, in_channels = x_q.shape

        # self-attention
        # [N, P, C] -> [N, P, 3C] -> [N, P, 3, h, c] where C = hc
        qkv = self.qkv_proj(x_q).reshape(b_sz, n_patches, 3, self.num_heads, -1)

        # [N, P, 3, h, c] -> [N, h, 3, P, C]
        qkv = qkv.transpose(1, 3).contiguous()

        # [N, h, 3, P, C] -> [N, h, P, C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [N h, P, c] -> [N, h, c, P]
        query = query.transpose(-1, -2)

        # Q^T*K
        # [N, h, c, P] x [N, h, P, c] -> [N, h, c, c]
        attn = torch.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, P, c] x [N, h, c, c] -> [N, h, P, c]
        out = torch.matmul(value,attn)

        # [N, h, P, c] -> [N, P, h, c] -> [N, P, C]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out
    
        # [N, P, C]
        b_sz, n_patches, in_channels = x_q.shape

        # self-attention
        # [N, P, C] batch无法化简，patch有很多化简方法了，C考虑ghost那种，从输入端就进行减少-> [N, P, 3C] -> [N, P, 3, h, c] where C = hc
        # 如果是分组shuffle的考虑，那首先比如分2组C/2,这一层参数少变成1/4.或者根据senet，选取高价值通道作为空间自注意力层的计算，这样就不局限于ghostnet，不局限于1/2
        qkv = self.qkv_proj(x_q).reshape(b_sz, n_patches, 3, self.num_heads, -1)

        # [N, P, 3, h, c] -> [N, h, 3, P, C]
        qkv = qkv.transpose(1, 3).contiguous()

        # [N, h, 3, P, C] -> [N, h, P, C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [N h, P, c] -> [N, h, c, P]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, P, c] x [N, h, c, P] -> [N, h, P, P]
        attn = torch.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, P, P] x [N, h, P, c] -> [N, h, P, c]
        out = torch.matmul(attn, value)

        # [N, h, P, c] -> [N, P, h, c] -> [N, P, C]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out


class TransformerEncoder(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        num_heads (int) : Number of heads in multi-head attention. Default: 8
        attn_dropout (float): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers. Default: 0.0

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        *args,
        **kwargs
    ) -> None:

        super().__init__()

        attn_unit = MultiHeadAttention(
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            bias=True
        )

        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        # multi-head attention
        res = x
        x = self.pre_norm_mha(x)
        x = x + res

        # feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""




def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
        deploy: Optional[bool] = False,
        is_last: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.is_last = is_last
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        self.deploy = deploy
        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )
        if deploy:
            self.conv_repararm = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=True
        )
        else:
            self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )
           
            
        if use_norm:
            self.norm = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)    
            
        if use_act:
            self.act = nn.SiLU()
        

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, 'reparam'):
            if self.act is not None:
                out = self.conv_repararm(x)
                if self.is_last:
                    return self.act(out),out
                return self.act(out)
            else:
                return self.conv_repararm(x)
        else:
            if hasattr(self, 'act'):
                if hasattr(self, 'norm'):
                    out=self.norm(self.conv(x))
                    if self.is_last:
                            return self.act(out),out
                    return self.act(out)
                else:
                    out=self.conv(x)
                    if self.is_last:
                        return self.act(out),out
                    return self.act(out)
            elif hasattr(self, 'norm'):
                if self.is_last:
                    out = self.norm(self.conv(x))
                    return out, out

                return self.norm(self.conv(x))
            if self.is_last:
                out = self.conv(x)
                return out, out
            return self.conv(x)


class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
        is_last: Optional[bool] = False
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()
        self.is_last = is_last
        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
                is_last = self.is_last,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            if self.is_last:
                out,preact=self.block(x)
                out=out +x
                preact=x + preact
                return out,preact
            return x + self.block(x)
        else:
            if self.is_last:
                out,preact = self.block(x)
                return out, preact
            return self.block(x)

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


# def conv_bn_rep(in_channels, out_channels, kernel_size, stride, padding, groups=1,use_bn=True,follow=False,pattern=3):
#     result = nn.Sequential()
#     if stride == 2 or in_channels!=out_channels:
#         result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                                       kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
#         # result.add_module('conv',DOConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,padding=padding,bias=False),)
#     else:
#         result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                                       kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
#         # result.add_module('conv',DOConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,padding=padding,bias=False),)
#     if use_bn:
#         result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
#     if follow:
#         result.add_module('conv1',RepTowerBlock(in_channels=out_channels,out_channels=out_channels,groups=groups,pattern=pattern,expand_rate=2))
#     return result


# class RepVGGBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False,use_act=True,use_bn=True,use_identity=True,count=True):
#         super(RepVGGBlock, self).__init__()
#         self.deploy = deploy
#         self.groups = groups
#         self.in_channels = in_channels
#         self.use_act = use_act
#         # self.count = 5 if count else 1

       
#         if self.use_act:
#             self.nonlinearity = nn.ReLU()
#         else:
#             self.nonlinearity = nn.Identity()

#         # if use_se:
#         #     #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
#         #     self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
#         # else:
#         self.se = nn.Identity()

#         if deploy:
#             self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                                       padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

#         else:
#             # self.rbr_identity_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
#             # self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 and use_identity else None
#             # self.dense_weight = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
#             self.rbr_dense = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=3)
#             # self.dense_weight_2 = Parameter(torch.Tensor(1,out_channels,1,1),requires_grad=True)
#             self.rbr_dense_2 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=2)
#             self.rbr_dense_3 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=True,pattern=1)
#             self.rbr_dense_4 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False)
#             self.rbr_dense_5 = conv_bn_rep(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups,use_bn=use_bn,follow=False)
            
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

#         # return self.nonlinearity(self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs))/2))
#         return self.nonlinearity(self.se((self.rbr_dense(inputs)+self.rbr_dense_2(inputs)+self.rbr_dense_3(inputs)+self.rbr_dense_4(inputs)+self.rbr_dense_5(inputs))/5))
#         # return self.nonlinearity(self.se(self.dense_weight*self.rbr_dense(inputs) + self.rbr_1x1_weight*self.rbr_1x1(inputs)+id_out))



# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio,count):
#         super(InvertedResidual, self).__init__()
#         self.blockname = None

#         self.stride = stride
#         assert stride in [1, 2]

#         self.use_res_connect = self.stride == 1 and inp == oup
#         if stride == 2:
#             self.pw = RepVGGBlock(in_channels=inp, out_channels=inp * expand_ratio, groups=1, kernel_size=1, stride=1, padding=0,count=count)
#         else:
#             self.pw = RepVGGBlock(in_channels=inp, out_channels=inp * expand_ratio, groups=1, kernel_size=1, stride=1, padding=0,count=count)
#         # self.pw_tower = RepTowerBlock(inp*expand_ratio,inp*expand_ratio,1,0,1,1,expand_rate=2)
#         # self.dw = RepVGGBlock(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio, groups=inp * expand_ratio, kernel_size=3, stride=stride, padding=1)
#         # self.pw_linear = nn.Sequential(nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
#         #         nn.BatchNorm2d(oup),)
#         self.dw = conv_bn_relu(inp*expand_ratio,inp*expand_ratio,3,stride,1,1,inp*expand_ratio)
#         # self.pw = conv_bn_relu(inp,inp*expand_ratio,1,1,0)

#         # if stride==2:
#         #     self.pw = nn.Sequential(
#         #         # pw
#         #         nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
#         #         # DOConv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, groups=1,padding=0,bias=False),
                
#         #         nn.BatchNorm2d(inp * expand_ratio),
#         #         # RepTowerBlock(inp * expand_ratio,inp * expand_ratio,expand_rate=2),
#         #         nn.ReLU(inplace=True),)
#         # else:
#         #     self.pw = nn.Sequential(
#         #         # pw
#         #         nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
#         #         # DOConv2d(inp, inp * expand_ratio, kernel_size=1, stride=1, groups=1,padding=0,bias=False),
                
#         #         nn.BatchNorm2d(inp * expand_ratio),
#         #         # RepTowerBlock(inp * expand_ratio,inp * expand_ratio,expand_rate=2),
#         #         nn.ReLU(inplace=True),)
#         #     # dw
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
#             # return self.conv(self.dw(self.pw(x)))
#             return self.conv(self.dw(self.pw(x)))


class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        head_dim: int = 32,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,
        is_last: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.is_last = is_last
        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            is_last = False
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)
        # if self.is_last:
        #     fm, preact = self.fusion(torch.cat((res, fm), dim=1))
        #     return fm, preact
        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


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

class MobileViT(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """
    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super().__init__()

        image_channels = 3
        out_channels = 16

        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            is_last=False
        )
        # self.conv_1 = Rep_bn(inchannels=image_channels,outchannels=out_channels,kernel_size=7,stride=2,group=1,bias=False)
        # 只有5*5和1*1效果不明显
        # self.conv_1_5_repadd = ConvLayer(
        #     in_channels=image_channels,
        #     out_channels=out_channels,
        #     kernel_size=5,
        #     stride=2,
        #     is_last=True
        # )
        # self.conv_1_7_repadd = ConvLayer(
        #     in_channels=image_channels,
        #     out_channels=out_channels,
        #     kernel_size=7,
        #     stride=2,
        #     is_last=True
        # )
        # self.conv_1_1_repadd = ConvLayer(
        #     in_channels=image_channels,
        #     out_channels=out_channels,
        #     kernel_size=1,
        #     stride=2,
        #     is_last=True
        # )
        self.act = nn.SiLU()

        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])

        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1,
            is_last = False
        )

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=num_classes))
        # self.classifier.add_module(name="fc",module=Classfier(exp_channels, num_classes, True))

        # weight init
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1
# (self, inp, oup, stride, expand_ratio,count):
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                # inp = input_channel,
                # oup = output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                # is_last= i == (num_blocks - 1)
                # count=True
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                # inp=input_channel,
                # oup=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                # count=True
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3,
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor,is_feat = False,preact=False) -> Tensor:
        first_down_pre_3 = self.conv_1(x)
        # x_1 , first_down_pre_1 = self.conv_1_1_repadd(x)
        # x_5 , first_down_pre_5 = self.conv_1_5_repadd(x)
        # x_7 , first_down_pre_7 = self.conv_1_7_repadd(x)
        # first_down_pre = first_down_pre_3+first_down_pre_1+first_down_pre_5+first_down_pre_7
        first_down = self.act(first_down_pre_3)
        # x, f1_pre = self.layer_1(first_down)
        # f1 = x
        # x, f2_pre = self.layer_2(x)
        # f2 = x
        # x, f3_pre = self.layer_3(x)
        # f3 = x
        # x, f4_pre = self.layer_4(x)
        # f4 = x
        # x, f5_pre = self.layer_5(x)
        # f5 = x
        # x , last_down_pre = self.conv_1x1_exp(x)

        x = self.layer_1(first_down)
        f1 = x
        x = self.layer_2(x)
        f2 = x
        x = self.layer_3(x)
        f3 = x
        x = self.layer_4(x)
        f4 = x
        x = self.layer_5(x)
        f5 = x
        x  = self.conv_1x1_exp(x)

        last_down = x
        out = self.classifier(x)
        # if is_feat:
        #     if preact:
        #         return [first_down_pre_3,f1_pre,f2_pre,f3_pre,f4_pre,f5_pre,last_down_pre],out
        #     else:
        #         return [first_down,f1,f2,f3,f4,f5,last_down],out
        # else:
        return out
            


        


def mobile_vit_xx_small_in7(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
    config = get_config("xx_small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit_x_small_in7(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt
    config = get_config("x_small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit_small_in7(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
    config = get_config("small")
    m = MobileViT(config, num_classes=num_classes)
    return m
def mobile_vit_tiny_in7(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
    config = get_config("tiny")
    m = MobileViT(config, num_classes=num_classes)
    return m

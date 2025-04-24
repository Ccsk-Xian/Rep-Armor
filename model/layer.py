import copy
from typing import Dict, List, Optional, OrderedDict, Tuple, Union
from torchvision.models import resnet
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _ntuple


from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn.modules.batchnorm import _BatchNorm




def build_norm(name: Optional[str], num_features: int) -> Optional[nn.Module]:
    if name is None:
        return None
    elif name == "bn_2d":
        return nn.BatchNorm2d(num_features)
    else:
        raise NotImplementedError


def build_act(name: Union[str, nn.Module, None]) -> Optional[nn.Module]:
    if name is None:
        return None
    elif isinstance(name, nn.Module):
        return name
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "relu6":
        return nn.ReLU6(inplace=True)
    elif name == "h_swish":
        return nn.Hardswish(inplace=True)
    elif name == "h_sigmoid":
        return nn.Hardsigmoid(inplace=True)
    else:
        raise NotImplementedError


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        padding = kernel_size // 2
        padding *= dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer_padding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        super(ConvLayer_padding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups


        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class PoolLayer(nn.Module):
    def __init__(
        self,
        pool_type='max',
        kernel_size=3,
        stride=2,
        padding=1,
    ):
        super(PoolLayer, self).__init__()
        if pool_type== 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return x


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout_rate=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.dropout = (
            nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0 else None
        )
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            for i in range(x.dim() - 1, 1, -1):
                x = torch.squeeze(x, dim=i)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class SELayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels=None,
        reduction=4,
        min_dim=16,
        act_func="relu",
    ):
        super(SELayer, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels or max(round(in_channels / reduction), min_dim)
        self.reduction = self.in_channels / self.mid_channels + 1e-10
        self.min_dim = min_dim

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.reduce_conv = nn.Conv2d(
            in_channels, self.mid_channels, kernel_size=(1, 1), bias=True
        )
        self.act = build_act(act_func)
        self.expand_conv = nn.Conv2d(
            self.mid_channels, in_channels, kernel_size=(1, 1), bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_attention = self.pooling(x)
        channel_attention = self.reduce_conv(channel_attention)
        channel_attention = self.act(channel_attention)
        channel_attention = self.expand_conv(channel_attention)
        channel_attention = F.hardsigmoid(channel_attention, inplace=True)
        return x * channel_attention


class DsConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        act_func=("relu6", None),
        norm=("bn_2d", "bn_2d"),
    ):
        super(DsConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class InvertedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
    ):
        super(InvertedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.expand_ratio = self.mid_channels / self.in_channels + 1e-10

        self.inverted_conv = ConvLayer(
            in_channels,
            self.mid_channels,
            1,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = ConvLayer(
            self.mid_channels,
            self.mid_channels,
            kernel_size,
            stride,
            groups=self.mid_channels,
            norm=norm[1],
            act_func=act_func[1],
        )
        self.point_conv = ConvLayer(
            self.mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class SeInvertedBlock(InvertedBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
        se_config: Optional[Dict] = None,
    ):
        super(SeInvertedBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            mid_channels=mid_channels,
            expand_ratio=expand_ratio,
            act_func=act_func,
            norm=norm,
        )
        se_config = se_config or {
            "reduction": 4,
            "min_dim": 16,
            "act_func": "relu",
        }
        self.se_layer = SELayer(self.depth_conv.out_channels, **se_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.se_layer(x)
        x = self.point_conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, conv: Optional[nn.Module], shortcut: Optional[nn.Module]):
        super(ResidualBlock, self).__init__()
        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            return x
        elif self.shortcut is None:
            return self.conv(x)
        else:
            return self.conv(x) + self.shortcut(x)

class BasicBlock_base(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(BasicBlock_base, self).__init__()
        self.conv1 = ConvLayer_padding(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, use_bias=False)
        self.conv2 = ConvLayer_padding(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, use_bias=False, act_func=None)
        self.relu = build_act('relu')
        if downsample:
            self.downsample = ConvLayer_padding(in_channels, out_channels, kernel_size=1, stride=stride, act_func=None)



    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.relu(x+identity)
        return x


class BasicBlock_baseV2(nn.Module):
    expansion = 1
    def __init__(self, conv1 : Optional[nn.Module], conv2 : Optional[nn.Module], downsample : Optional[nn.Module]):
        super(BasicBlock_baseV2, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.relu = build_act('relu')
        self.downsample = downsample

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        #print(x.shape, identity.shape)
        x = self.relu(x + identity)
        return x


class OpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x




def make_divisible(
    v: Union[int, float], divisor: Optional[int], min_val=None
) -> Union[int, float]:
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if divisor is None:
        return v

    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def load_state_dict_from_file(file: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(file, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def list_sum(x: List) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: List) -> Any:
    return list_sum(x) / len(x)


def parse_unknown_args(unknown: List) -> Dict:
    """Parse unknown args."""
    index = 0
    parsed_dict = {}
    while index < len(unknown):
        key, val = unknown[index], unknown[index + 1]
        index += 2
        if key.startswith("--"):
            key = key[2:]
            try:
                # try parsing with yaml
                if "{" in val and "}" in val and ":" in val:
                    val = val.replace(":", ": ")  # add space manually for dict
                out_val = yaml.safe_load(val)
            except ValueError:
                # return raw string if parsing fails
                out_val = val
            parsed_dict[key] = out_val
    return parsed_dict


def partial_update_config(config: Dict, partial_config: Dict):
    for key in partial_config:
        if (
            key in config
            and isinstance(partial_config[key], Dict)
            and isinstance(config[key], Dict)
        ):
            partial_update_config(config[key], partial_config[key])
        else:
            config[key] = partial_config[key]


def remove_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.weight = m.bias = None
            m.forward = lambda x: x


def get_same_padding(kernel_size: Union[int, Tuple[int, int]]) -> Union[int, tuple]:
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, f"invalid kernel size: {kernel_size}"
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    else:
        assert isinstance(
            kernel_size, int
        ), "kernel size should be either `int` or `tuple`"
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def torch_random_choices(
    src_list: List[Any],
    generator: Optional[torch.Generator],
    k=1,
) -> Union[Any, List[Any]]:
    rand_idx = torch.randint(low=0, high=len(src_list), generator=generator, size=(k,))
    out_list = [src_list[i] for i in rand_idx]
    return out_list[0] if k == 1 else out_list




def build_norm(name: Optional[str], num_features: int) -> Optional[nn.Module]:
    if name is None:
        return None
    elif name == "bn_2d":
        return DynamicBatchNorm2d(num_features)
    else:
        raise NotImplementedError


class DynamicModule(nn.Module):
    def export(self) -> nn.Module:
        #print('export')
        raise NotImplementedError

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = self.state_dict()
        for prefix, module in self.named_children():
            if isinstance(module, DynamicModule):
                for name, tensor in module.active_state_dict().items():
                    state_dict[prefix + "." + name] = tensor
        return state_dict


class DynamicConv2d(DynamicModule, nn.Conv2d):
    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        bias: bool = True,
    ) -> None:
        kernel_size = _ntuple(self._ndim)(kernel_size)
        stride = _ntuple(self._ndim)(stride)
        dilation = _ntuple(self._ndim)(dilation)
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=_ntuple(self._ndim)(0),
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode="zeros",
        )
        self.active_in_channels = in_channels
        self.active_out_channels = out_channels

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_out_channels, : self.active_in_channels]
        return weight.contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_out_channels].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_channels = x.shape[1]
        if self.padding_mode != "zeros":
            raise NotImplementedError
        else:
            active_weight = self.active_weight
            return getattr(F, "conv{}d".format(self._ndim))(
                x,
                active_weight,
                self.active_bias,
                stride=self.stride,
                padding=get_same_padding(int(active_weight.size(2))) * self.dilation[0],
                dilation=self.dilation,
                groups=1,
            )

    def export(self) -> nn.Module:
        module = getattr(nn, "Conv{}d".format(self._ndim))(
            self.active_in_channels,
            self.active_out_channels,
            self.kernel_size[0],
            stride=self.stride,
            padding=get_same_padding(self.kernel_size[0]) * self.dilation[0],
            dilation=self.dilation,
            groups=1,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict

class DynamicConv2d_padding(DynamicModule, nn.Conv2d):
    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding:int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        bias: bool = True,
    ) -> None:
        kernel_size = _ntuple(self._ndim)(kernel_size)
        stride = _ntuple(self._ndim)(stride)
        dilation = _ntuple(self._ndim)(dilation)
        self.padding_n = padding
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode="zeros",
        )
        self.active_in_channels = in_channels
        self.active_out_channels = out_channels

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_out_channels, : self.active_in_channels]
        return weight.contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_out_channels].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_channels = x.shape[1]
        if self.padding_mode != "zeros":
            raise NotImplementedError
        else:
            active_weight = self.active_weight
            return getattr(F, "conv{}d".format(self._ndim))(
                x,
                active_weight,
                self.active_bias,
                stride=self.stride,
                padding=self.padding_n,
                dilation=self.dilation,
                groups=1,
            )

    def export(self) -> nn.Module:
        module = getattr(nn, "Conv{}d".format(self._ndim))(
            self.active_in_channels,
            self.active_out_channels,
            self.kernel_size[0],
            stride=self.stride,
            padding=self.padding_n,
            dilation=self.dilation,
            groups=1,
            bias=self.bias is not None,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicDepthwiseConv2d(DynamicModule, nn.Conv2d):
    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        bias: bool = True,
    ) -> None:
        kernel_size = _ntuple(self._ndim)(kernel_size)
        stride = _ntuple(self._ndim)(stride)
        dilation = _ntuple(self._ndim)(dilation)
        nn.Conv2d.__init__(
            self,
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=_ntuple(self._ndim)(0),
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode="zeros",
        )
        self.active_in_channels = in_channels

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_in_channels, : self.active_in_channels]
        return weight.contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_in_channels].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_channels = x.shape[1]
        if self.padding_mode != "zeros":
            raise NotImplementedError
        else:
            active_weight = self.active_weight
            return getattr(F, "conv{}d".format(self._ndim))(
                x,
                active_weight,
                self.active_bias,
                stride=self.stride,
                padding=get_same_padding(int(active_weight.size(2))) * self.dilation[0],
                dilation=self.dilation,
                groups=self.active_in_channels,
            )

    def export(self) -> nn.Module:
        module = getattr(nn, "Conv{}d".format(self._ndim))(
            self.active_in_channels,
            self.active_in_channels,
            self.kernel_size[0],
            stride=self.stride,
            padding=get_same_padding(self.kernel_size[0]) * self.dilation[0],
            dilation=self.dilation,
            groups=self.active_in_channels,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicLinear(nn.Linear, DynamicModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.active_in_features = in_features
        self.active_out_features = out_features

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        return self.weight[
            : self.active_out_features, : self.active_in_features
        ].contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_out_features].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_features = x.shape[-1]
        return F.linear(x, weight=self.active_weight, bias=self.active_bias)

    def export(self) -> nn.Module:
        module = nn.Linear(
            self.active_in_features,
            self.active_out_features,
            bias=self.bias is not None,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicBatchNorm2d(DynamicModule, nn.BatchNorm2d):
    _ndim = 2

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        nn.BatchNorm2d.__init__(
            self,
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.active_num_features = num_features

    @property
    def active_running_mean(self) -> Optional[torch.Tensor]:
        if self.running_mean is None:
            return None
        return self.running_mean[: self.active_num_features]

    @property
    def active_running_var(self) -> Optional[torch.Tensor]:
        if self.running_var is None:
            return None
        return self.running_var[: self.active_num_features]

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        return self.weight[: self.active_num_features]

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_num_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        self.active_num_features = x.shape[1]

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.active_running_mean is None) and (
                self.active_running_var is None
            )

        running_mean = (
            self.active_running_mean
            if not self.training or self.track_running_stats
            else None
        )
        running_var = (
            self.active_running_var
            if not self.training or self.track_running_stats
            else None
        )

        return F.batch_norm(
            x,
            running_mean,
            running_var,
            self.active_weight,
            self.active_bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def export(self) -> nn.Module:
        module = getattr(nn, "BatchNorm{}d".format(self._ndim))(
            self.active_num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.running_mean is not None:
            state_dict["running_mean"] = self.active_running_mean
        if self.running_var is not None:
            state_dict["running_var"] = self.active_running_var
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicConvLayer(ConvLayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        nn.Module.__init__(self)
        self.conv = DynamicConv2d(
            in_channels=in_channels,
            out_channels=max(out_channels),
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias,
        )
        self.norm = build_norm(norm, max(out_channels))
        self.act = build_act(act_func)

        self.in_channels = in_channels
        self.out_channels_list = copy.deepcopy(out_channels)

    def export(self) -> ConvLayer:
        module = ConvLayer.__new__(ConvLayer)
        nn.Module.__init__(module)
        module.conv = self.conv.export()
        module.norm = (
            self.norm.export() if isinstance(self.norm, DynamicModule) else self.norm
        )
        module.act = self.act
        return module

class DynamicConvLayer_padding(ConvLayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        nn.Module.__init__(self)
        self.conv = DynamicConv2d_padding(
            in_channels=in_channels,
            out_channels=max(out_channels),
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias,
            padding=padding,
        )
        self.norm = build_norm(norm, max(out_channels))
        self.act = build_act(act_func)

        self.in_channels = in_channels
        self.out_channels_list = copy.deepcopy(out_channels)

    def export(self) -> ConvLayer:
        module = ConvLayer.__new__(ConvLayer)
        #print(isinstance(self.norm, DynamicModule))
        nn.Module.__init__(module)
        module.conv = self.conv.export()
        module.norm = (
            self.norm.export() if isinstance(self.norm, DynamicModule) else self.norm
        )
        module.act = self.act
        return module



class DynamicDepthwiseConvLayer(DynamicConvLayer):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride=1,
        dilation=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        nn.Module.__init__(self)
        self.conv = DynamicDepthwiseConv2d(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias,
        )
        self.norm = build_norm(norm, in_channels)
        self.act = build_act(act_func)


class DynamicLinearLayer(LinearLayer, DynamicModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout_rate=0,
        norm=None,
        act_func=None,
    ):
        DynamicModule.__init__(self)

        self.dropout = (
            nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0 else None
        )
        self.linear = DynamicLinear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, out_features)
        self.act = build_act(act_func)

    def export(self) -> LinearLayer:
        module = LinearLayer.__new__(LinearLayer)
        nn.Module.__init__(module)
        module.dropout = self.dropout
        module.linear = self.linear.export()
        module.norm = (
            self.norm.export() if isinstance(self.norm, DynamicModule) else self.norm
        )
        module.act = self.act
        return module


class DynamicSE(SELayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        mid_channels=None,
        reduction=4,
        min_dim=16,
        act_func="relu",
    ):
        DynamicModule.__init__(self)
        self.min_dim = min_dim

        if mid_channels is None:
            mid_channels = max(round(in_channels / reduction), min_dim)
        self.reduction = in_channels / mid_channels

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.reduce_conv = DynamicConv2d(
            in_channels, mid_channels, kernel_size=1, bias=True
        )
        self.act = build_act(act_func)
        self.expand_conv = DynamicConv2d(
            mid_channels, in_channels, kernel_size=1, bias=True
        )

        self.active_in_channels = in_channels

    @property
    def active_mid_channels(self):
        return make_divisible(
            max(self.active_in_channels / self.reduction, self.min_dim), 1
        )

    def forward(self, x):
        self.active_in_channels = x.shape[1]
        self.reduce_conv.active_in_channels = self.active_in_channels
        self.reduce_conv.active_out_channels = self.active_mid_channels
        self.expand_conv.active_in_channels = self.active_mid_channels
        self.expand_conv.active_out_channels = self.active_in_channels

        return SELayer.forward(self, x)

    def export(self) -> SELayer:
        module = SELayer(
            in_channels=self.active_in_channels,
            mid_channels=self.active_mid_channels,
        )
        module.act = self.act
        module.load_state_dict(self.active_state_dict())
        return module


class DynamicDsConvLayer(DsConvLayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size=3,
        stride=1,
        act_func=("relu6", None),
        norm=("bn_2d", "bn_2d"),
    ):
        nn.Module.__init__(self)
        self.depth_conv = DynamicDepthwiseConvLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = DynamicConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm[1],
            act_func=act_func[1],
        )

    def export(self) -> DsConvLayer:
        module = DsConvLayer.__new__(DsConvLayer)
        nn.Module.__init__(module)
        module.depth_conv = self.depth_conv.export()
        module.point_conv = self.point_conv.export()
        return module


class DynamicInvertedBlock(InvertedBlock, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size: int,
        expand_ratio: List[float],
        stride=1,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
    ):
        nn.Module.__init__(self)

        mid_channels = make_divisible(in_channels * max(expand_ratio), 1)

        self.inverted_conv = DynamicConvLayer(
            in_channels=in_channels,
            out_channels=[mid_channels],
            kernel_size=1,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = DynamicDepthwiseConvLayer(
            in_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm[1],
            act_func=act_func[1],
        )
        self.point_conv = DynamicConvLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm[2],
            act_func=act_func[2],
        )

        self.expand_ratio_list = copy.deepcopy(expand_ratio)

    def export(self) -> InvertedBlock:
        module = InvertedBlock.__new__(InvertedBlock)
        nn.Module.__init__(module)
        module.inverted_conv = self.inverted_conv.export()
        module.depth_conv = self.depth_conv.export()
        module.point_conv = self.point_conv.export()
        return module


class DynamicSeInvertedBlock(SeInvertedBlock, DynamicInvertedBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size: int,
        expand_ratio: List[float],
        stride=1,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
        se_config: Optional[Dict] = None,
    ):
        DynamicInvertedBlock.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio,
            stride=stride,
            act_func=act_func,
            norm=norm,
        )

        se_config = {} or se_config
        self.se_layer = DynamicSE(in_channels=self.point_conv.in_channels, **se_config)

    def export(self) -> SeInvertedBlock:
        module = SeInvertedBlock.__new__(SeInvertedBlock)
        nn.Module.__init__(module)
        module.inverted_conv = self.inverted_conv.export()
        module.depth_conv = self.depth_conv.export()
        module.se_layer = self.se_layer.export()
        module.point_conv = self.point_conv.export()
        return module


class DynamicBasicBlock_base(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(DynamicBasicBlock_base, self).__init__()
        self.conv1 = DynamicConvLayer_padding(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, use_bias=False)
        self.conv2 = DynamicConvLayer_padding(max(out_channels), out_channels, kernel_size=3, padding=1, stride=stride, use_bias=False, act_func=None)
        self.relu = build_act('relu')
        if downsample:
            self.downsample = DynamicConvLayer_padding(in_channels, out_channels, kernel_size=1, stride=stride, act_func=None)
        else:
            self.downsample = None


    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.relu(x+identity)
        return x
    



def build_norm(name: Optional[str], num_features: int) -> Optional[nn.Module]:
    if name is None:
        return None
    elif name == "bn_2d":
        return DynamicBatchNorm2d(num_features)
    else:
        raise NotImplementedError


class DynamicModule(nn.Module):
    def export(self) -> nn.Module:
        #print('export')
        raise NotImplementedError

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = self.state_dict()
        for prefix, module in self.named_children():
            if isinstance(module, DynamicModule):
                for name, tensor in module.active_state_dict().items():
                    state_dict[prefix + "." + name] = tensor
        return state_dict


class DynamicConv2d(DynamicModule, nn.Conv2d):
    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        bias: bool = True,
    ) -> None:
        kernel_size = _ntuple(self._ndim)(kernel_size)
        stride = _ntuple(self._ndim)(stride)
        dilation = _ntuple(self._ndim)(dilation)
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=_ntuple(self._ndim)(0),
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode="zeros",
        )
        self.active_in_channels = in_channels
        self.active_out_channels = out_channels

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_out_channels, : self.active_in_channels]
        return weight.contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_out_channels].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_channels = x.shape[1]
        if self.padding_mode != "zeros":
            raise NotImplementedError
        else:
            active_weight = self.active_weight
            return getattr(F, "conv{}d".format(self._ndim))(
                x,
                active_weight,
                self.active_bias,
                stride=self.stride,
                padding=get_same_padding(int(active_weight.size(2))) * self.dilation[0],
                dilation=self.dilation,
                groups=1,
            )

    def export(self) -> nn.Module:
        module = getattr(nn, "Conv{}d".format(self._ndim))(
            self.active_in_channels,
            self.active_out_channels,
            self.kernel_size[0],
            stride=self.stride,
            padding=get_same_padding(self.kernel_size[0]) * self.dilation[0],
            dilation=self.dilation,
            groups=1,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict

class DynamicConv2d_padding(DynamicModule, nn.Conv2d):
    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding:int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        bias: bool = True,
    ) -> None:
        kernel_size = _ntuple(self._ndim)(kernel_size)
        stride = _ntuple(self._ndim)(stride)
        dilation = _ntuple(self._ndim)(dilation)
        self.padding_n = padding
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode="zeros",
        )
        self.active_in_channels = in_channels
        self.active_out_channels = out_channels

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_out_channels, : self.active_in_channels]
        return weight.contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_out_channels].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_channels = x.shape[1]
        if self.padding_mode != "zeros":
            raise NotImplementedError
        else:
            active_weight = self.active_weight
            return getattr(F, "conv{}d".format(self._ndim))(
                x,
                active_weight,
                self.active_bias,
                stride=self.stride,
                padding=self.padding_n,
                dilation=self.dilation,
                groups=1,
            )

    def export(self) -> nn.Module:
        module = getattr(nn, "Conv{}d".format(self._ndim))(
            self.active_in_channels,
            self.active_out_channels,
            self.kernel_size[0],
            stride=self.stride,
            padding=self.padding_n,
            dilation=self.dilation,
            groups=1,
            bias=self.bias is not None,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicDepthwiseConv2d(DynamicModule, nn.Conv2d):
    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        bias: bool = True,
    ) -> None:
        kernel_size = _ntuple(self._ndim)(kernel_size)
        stride = _ntuple(self._ndim)(stride)
        dilation = _ntuple(self._ndim)(dilation)
        nn.Conv2d.__init__(
            self,
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=_ntuple(self._ndim)(0),
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode="zeros",
        )
        self.active_in_channels = in_channels

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_in_channels, : self.active_in_channels]
        return weight.contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_in_channels].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_channels = x.shape[1]
        if self.padding_mode != "zeros":
            raise NotImplementedError
        else:
            active_weight = self.active_weight
            return getattr(F, "conv{}d".format(self._ndim))(
                x,
                active_weight,
                self.active_bias,
                stride=self.stride,
                padding=get_same_padding(int(active_weight.size(2))) * self.dilation[0],
                dilation=self.dilation,
                groups=self.active_in_channels,
            )

    def export(self) -> nn.Module:
        module = getattr(nn, "Conv{}d".format(self._ndim))(
            self.active_in_channels,
            self.active_in_channels,
            self.kernel_size[0],
            stride=self.stride,
            padding=get_same_padding(self.kernel_size[0]) * self.dilation[0],
            dilation=self.dilation,
            groups=self.active_in_channels,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicLinear(nn.Linear, DynamicModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.active_in_features = in_features
        self.active_out_features = out_features

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        return self.weight[
            : self.active_out_features, : self.active_in_features
        ].contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_out_features].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_features = x.shape[-1]
        return F.linear(x, weight=self.active_weight, bias=self.active_bias)

    def export(self) -> nn.Module:
        module = nn.Linear(
            self.active_in_features,
            self.active_out_features,
            bias=self.bias is not None,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicBatchNorm2d(DynamicModule, nn.BatchNorm2d):
    _ndim = 2

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        nn.BatchNorm2d.__init__(
            self,
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.active_num_features = num_features

    @property
    def active_running_mean(self) -> Optional[torch.Tensor]:
        if self.running_mean is None:
            return None
        return self.running_mean[: self.active_num_features]

    @property
    def active_running_var(self) -> Optional[torch.Tensor]:
        if self.running_var is None:
            return None
        return self.running_var[: self.active_num_features]

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        return self.weight[: self.active_num_features]

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_num_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        self.active_num_features = x.shape[1]

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.active_running_mean is None) and (
                self.active_running_var is None
            )

        running_mean = (
            self.active_running_mean
            if not self.training or self.track_running_stats
            else None
        )
        running_var = (
            self.active_running_var
            if not self.training or self.track_running_stats
            else None
        )

        return F.batch_norm(
            x,
            running_mean,
            running_var,
            self.active_weight,
            self.active_bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def export(self) -> nn.Module:
        module = getattr(nn, "BatchNorm{}d".format(self._ndim))(
            self.active_num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.running_mean is not None:
            state_dict["running_mean"] = self.active_running_mean
        if self.running_var is not None:
            state_dict["running_var"] = self.active_running_var
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicConvLayer(ConvLayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        nn.Module.__init__(self)
        self.conv = DynamicConv2d(
            in_channels=in_channels,
            out_channels=max(out_channels),
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias,
        )
        self.norm = build_norm(norm, max(out_channels))
        self.act = build_act(act_func)

        self.in_channels = in_channels
        self.out_channels_list = copy.deepcopy(out_channels)

    def export(self) -> ConvLayer:
        module = ConvLayer.__new__(ConvLayer)
        nn.Module.__init__(module)
        module.conv = self.conv.export()
        module.norm = (
            self.norm.export() if isinstance(self.norm, DynamicModule) else self.norm
        )
        module.act = self.act
        return module

class DynamicConvLayer_padding(ConvLayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        nn.Module.__init__(self)
        self.conv = DynamicConv2d_padding(
            in_channels=in_channels,
            out_channels=max(out_channels),
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias,
            padding=padding,
        )
        self.norm = build_norm(norm, max(out_channels))
        self.act = build_act(act_func)

        self.in_channels = in_channels
        self.out_channels_list = copy.deepcopy(out_channels)

    def export(self) -> ConvLayer:
        module = ConvLayer.__new__(ConvLayer)
        #print(isinstance(self.norm, DynamicModule))
        nn.Module.__init__(module)
        module.conv = self.conv.export()
        module.norm = (
            self.norm.export() if isinstance(self.norm, DynamicModule) else self.norm
        )
        module.act = self.act
        return module



class DynamicDepthwiseConvLayer(DynamicConvLayer):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride=1,
        dilation=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        nn.Module.__init__(self)
        self.conv = DynamicDepthwiseConv2d(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias,
        )
        self.norm = build_norm(norm, in_channels)
        self.act = build_act(act_func)


class DynamicLinearLayer(LinearLayer, DynamicModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout_rate=0,
        norm=None,
        act_func=None,
    ):
        DynamicModule.__init__(self)

        self.dropout = (
            nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0 else None
        )
        self.linear = DynamicLinear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, out_features)
        self.act = build_act(act_func)

    def export(self) -> LinearLayer:
        module = LinearLayer.__new__(LinearLayer)
        nn.Module.__init__(module)
        module.dropout = self.dropout
        module.linear = self.linear.export()
        module.norm = (
            self.norm.export() if isinstance(self.norm, DynamicModule) else self.norm
        )
        module.act = self.act
        return module


class DynamicSE(SELayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        mid_channels=None,
        reduction=4,
        min_dim=16,
        act_func="relu",
    ):
        DynamicModule.__init__(self)
        self.min_dim = min_dim

        if mid_channels is None:
            mid_channels = max(round(in_channels / reduction), min_dim)
        self.reduction = in_channels / mid_channels

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.reduce_conv = DynamicConv2d(
            in_channels, mid_channels, kernel_size=1, bias=True
        )
        self.act = build_act(act_func)
        self.expand_conv = DynamicConv2d(
            mid_channels, in_channels, kernel_size=1, bias=True
        )

        self.active_in_channels = in_channels

    @property
    def active_mid_channels(self):
        return make_divisible(
            max(self.active_in_channels / self.reduction, self.min_dim), 1
        )

    def forward(self, x):
        self.active_in_channels = x.shape[1]
        self.reduce_conv.active_in_channels = self.active_in_channels
        self.reduce_conv.active_out_channels = self.active_mid_channels
        self.expand_conv.active_in_channels = self.active_mid_channels
        self.expand_conv.active_out_channels = self.active_in_channels

        return SELayer.forward(self, x)

    def export(self) -> SELayer:
        module = SELayer(
            in_channels=self.active_in_channels,
            mid_channels=self.active_mid_channels,
        )
        module.act = self.act
        module.load_state_dict(self.active_state_dict())
        return module


class DynamicDsConvLayer(DsConvLayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size=3,
        stride=1,
        act_func=("relu6", None),
        norm=("bn_2d", "bn_2d"),
    ):
        nn.Module.__init__(self)
        self.depth_conv = DynamicDepthwiseConvLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = DynamicConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm[1],
            act_func=act_func[1],
        )

    def export(self) -> DsConvLayer:
        module = DsConvLayer.__new__(DsConvLayer)
        nn.Module.__init__(module)
        module.depth_conv = self.depth_conv.export()
        module.point_conv = self.point_conv.export()
        return module


class DynamicInvertedBlock(InvertedBlock, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size: int,
        expand_ratio: List[float],
        stride=1,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
    ):
        nn.Module.__init__(self)

        mid_channels = make_divisible(in_channels * max(expand_ratio), 1)

        self.inverted_conv = DynamicConvLayer(
            in_channels=in_channels,
            out_channels=[mid_channels],
            kernel_size=1,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = DynamicDepthwiseConvLayer(
            in_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm[1],
            act_func=act_func[1],
        )
        self.point_conv = DynamicConvLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm[2],
            act_func=act_func[2],
        )

        self.expand_ratio_list = copy.deepcopy(expand_ratio)

    def export(self) -> InvertedBlock:
        module = InvertedBlock.__new__(InvertedBlock)
        nn.Module.__init__(module)
        module.inverted_conv = self.inverted_conv.export()
        module.depth_conv = self.depth_conv.export()
        module.point_conv = self.point_conv.export()
        return module


class DynamicSeInvertedBlock(SeInvertedBlock, DynamicInvertedBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size: int,
        expand_ratio: List[float],
        stride=1,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
        se_config: Optional[Dict] = None,
    ):
        DynamicInvertedBlock.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio,
            stride=stride,
            act_func=act_func,
            norm=norm,
        )

        se_config = {} or se_config
        self.se_layer = DynamicSE(in_channels=self.point_conv.in_channels, **se_config)

    def export(self) -> SeInvertedBlock:
        module = SeInvertedBlock.__new__(SeInvertedBlock)
        nn.Module.__init__(module)
        module.inverted_conv = self.inverted_conv.export()
        module.depth_conv = self.depth_conv.export()
        module.se_layer = self.se_layer.export()
        module.point_conv = self.point_conv.export()
        return module


class DynamicBasicBlock_base(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(DynamicBasicBlock_base, self).__init__()
        self.conv1 = DynamicConvLayer_padding(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, use_bias=False)
        self.conv2 = DynamicConvLayer_padding(max(out_channels), out_channels, kernel_size=3, padding=1, stride=stride, use_bias=False, act_func=None)
        self.relu = build_act('relu')
        if downsample:
            self.downsample = DynamicConvLayer_padding(in_channels, out_channels, kernel_size=1, stride=stride, act_func=None)
        else:
            self.downsample = None


    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.relu(x+identity)
        return x
    



def reset_bn(
        model: nn.Module, data_loader, sync=False, backend="ddp", progress_bar=False
) -> None:
    bn_mean = {}
    bn_var = {}

    tmp_model = copy.deepcopy(model)
    for name, m in tmp_model.named_modules():
        if isinstance(m, _BatchNorm):
            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    x = x.contiguous()
                    if sync:
                        batch_mean = (
                            x.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )  # 1, C, 1, 1
                        if backend == "ddp":
                            batch_mean = ddp_reduce_tensor(batch_mean, reduce="cat")
                        else:
                            raise NotImplementedError
                        batch_mean = torch.mean(batch_mean, dim=0, keepdim=True)

                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = (
                            batch_var.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )
                        if backend == "ddp":
                            batch_var = ddp_reduce_tensor(batch_var, reduce="cat")
                        else:
                            raise NotImplementedError
                        batch_var = torch.mean(batch_var, dim=0, keepdim=True)
                    else:
                        batch_mean = (
                            x.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = (
                            batch_var.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.shape[0]
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    # skip if there is no batch normalization layers in the network
    if len(bn_mean) == 0:
        return

    tmp_model.eval()
    with torch.no_grad():
        with tqdm(
                total=len(data_loader), desc="reset bn", disable=(not progress_bar)
        ) as t:
            for images, _ in data_loader:
                images = images.cuda()
                tmp_model(images)
                t.set_postfix(
                    {
                        "batch_size": images.size(0),
                        "image_size": images.size(2),
                    }
                )
                t.update()

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, _BatchNorm)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def aug_width(
        base_width: float, factor_list: List[float], divisor: Optional[int] = None
) -> List[Union[float, int]]:
    out_list = [base_width * factor for factor in factor_list]
    if divisor is not None:
        out_list = [make_divisible(out_dim, divisor) for out_dim in out_list]
    return out_list


def sync_width(width) -> int:
    width = ddp_reduce_tensor(torch.Tensor(1).fill_(width).cuda(), "root")
    return int(width)


def sort_param(
        param: nn.Parameter,
        dim: int,
        sorted_idx: torch.Tensor,
) -> None:
    param.data.copy_(
        torch.clone(torch.index_select(param.data, dim, sorted_idx)).detach()
    )


def sort_norm(norm, sorted_idx: torch.Tensor) -> None:
    sort_param(norm.weight, 0, sorted_idx)
    sort_param(norm.bias, 0, sorted_idx)
    try:
        sort_param(norm.running_mean, 0, sorted_idx)
        sort_param(norm.running_var, 0, sorted_idx)
    except AttributeError:
        pass


def sort_se(se: SELayer, sorted_idx: torch.Tensor) -> None:
    # expand conv, output dim 0
    sort_param(se.expand_conv.weight, 0, sorted_idx)
    sort_param(se.expand_conv.bias, 0, sorted_idx)
    # reduce conv, input dim 1
    sort_param(se.reduce_conv.weight, 1, sorted_idx)

    # sort middle weight
    importance = torch.sum(torch.abs(se.expand_conv.weight.data), dim=(0, 2, 3))
    sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
    # expand conv, input dim 1
    sort_param(se.expand_conv.weight, 1, sorted_idx)
    # reduce conv, output dim 0
    sort_param(se.reduce_conv.weight, 0, sorted_idx)
    sort_param(se.reduce_conv.bias, 0, sorted_idx)


def sort_channels_inner(block) -> None:
    if isinstance(block, (InvertedBlock, SeInvertedBlock)):
        # calc channel importance
        importance = torch.sum(
            torch.abs(block.point_conv.conv.weight.data), dim=(0, 2, 3)
        )
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        # sort based on sorted_idx
        sort_param(block.point_conv.conv.weight, 1, sorted_idx)
        sort_norm(block.depth_conv.norm, sorted_idx)
        sort_param(block.depth_conv.conv.weight, 0, sorted_idx)
        sort_norm(block.inverted_conv.norm, sorted_idx)
        sort_param(block.inverted_conv.conv.weight, 0, sorted_idx)
        if isinstance(block, SeInvertedBlock):
            sort_se(block.se_layer, sorted_idx)
    else:
        raise NotImplementedError

def sort_channels_resblock(block) -> None:
    #print(block)
    if isinstance(block, (DynamicConvLayer_padding)):
        # calc channel importance
        importance = torch.sum(
            torch.abs(block.conv.weight.data), dim=(0, 2, 3)
        )
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        # sort based on sorted_idx
        if block.conv.in_channels == block.conv.out_channels:
            sort_param(block.conv.weight, 1, sorted_idx)
            sort_norm(block.norm, sorted_idx)
        else:
            sort_param(block.conv.weight, 1, sorted_idx)
    else:
        raise NotImplementedError
    



# def build_norm(name: Optional[str], num_features: int) -> Optional[nn.Module]:
#     if name is None:
#         return None
#     elif name == "bn_2d":
#         return nn.BatchNorm2d(num_features)
#     else:
#         raise NotImplementedError


def build_act(name: Union[str, nn.Module, None]) -> Optional[nn.Module]:
    if name is None:
        return None
    elif isinstance(name, nn.Module):
        return name
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "relu6":
        return nn.ReLU6(inplace=True)
    elif name == "h_swish":
        return nn.Hardswish(inplace=True)
    elif name == "h_sigmoid":
        return nn.Hardsigmoid(inplace=True)
    else:
        raise NotImplementedError


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        padding = kernel_size // 2
        padding *= dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer_padding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        super(ConvLayer_padding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups


        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class PoolLayer(nn.Module):
    def __init__(
        self,
        pool_type='max',
        kernel_size=3,
        stride=2,
        padding=1,
    ):
        super(PoolLayer, self).__init__()
        if pool_type== 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return x


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout_rate=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.dropout = (
            nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0 else None
        )
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            for i in range(x.dim() - 1, 1, -1):
                x = torch.squeeze(x, dim=i)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class SELayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels=None,
        reduction=4,
        min_dim=16,
        act_func="relu",
    ):
        super(SELayer, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels or max(round(in_channels / reduction), min_dim)
        self.reduction = self.in_channels / self.mid_channels + 1e-10
        self.min_dim = min_dim

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.reduce_conv = nn.Conv2d(
            in_channels, self.mid_channels, kernel_size=(1, 1), bias=True
        )
        self.act = build_act(act_func)
        self.expand_conv = nn.Conv2d(
            self.mid_channels, in_channels, kernel_size=(1, 1), bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_attention = self.pooling(x)
        channel_attention = self.reduce_conv(channel_attention)
        channel_attention = self.act(channel_attention)
        channel_attention = self.expand_conv(channel_attention)
        channel_attention = F.hardsigmoid(channel_attention, inplace=True)
        return x * channel_attention


class DsConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        act_func=("relu6", None),
        norm=("bn_2d", "bn_2d"),
    ):
        super(DsConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class InvertedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
    ):
        super(InvertedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.expand_ratio = self.mid_channels / self.in_channels + 1e-10

        self.inverted_conv = ConvLayer(
            in_channels,
            self.mid_channels,
            1,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = ConvLayer(
            self.mid_channels,
            self.mid_channels,
            kernel_size,
            stride,
            groups=self.mid_channels,
            norm=norm[1],
            act_func=act_func[1],
        )
        self.point_conv = ConvLayer(
            self.mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class SeInvertedBlock(InvertedBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
        se_config: Optional[Dict] = None,
    ):
        super(SeInvertedBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            mid_channels=mid_channels,
            expand_ratio=expand_ratio,
            act_func=act_func,
            norm=norm,
        )
        se_config = se_config or {
            "reduction": 4,
            "min_dim": 16,
            "act_func": "relu",
        }
        self.se_layer = SELayer(self.depth_conv.out_channels, **se_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.se_layer(x)
        x = self.point_conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, conv: Optional[nn.Module], shortcut: Optional[nn.Module]):
        super(ResidualBlock, self).__init__()
        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            return x
        elif self.shortcut is None:
            return self.conv(x)
        else:
            return self.conv(x) + self.shortcut(x)

class BasicBlock_base(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(BasicBlock_base, self).__init__()
        self.conv1 = ConvLayer_padding(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, use_bias=False)
        self.conv2 = ConvLayer_padding(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, use_bias=False, act_func=None)
        self.relu = build_act('relu')
        if downsample:
            self.downsample = ConvLayer_padding(in_channels, out_channels, kernel_size=1, stride=stride, act_func=None)



    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.relu(x+identity)
        return x


class BasicBlock_baseV2(nn.Module):
    expansion = 1
    def __init__(self, conv1 : Optional[nn.Module], conv2 : Optional[nn.Module], downsample : Optional[nn.Module]):
        super(BasicBlock_baseV2, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.relu = build_act('relu')
        self.downsample = downsample

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        #print(x.shape, identity.shape)
        x = self.relu(x + identity)
        return x


class OpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
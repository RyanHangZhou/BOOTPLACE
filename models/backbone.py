import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List

import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d with fixed batch statistics and affine parameters.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        state_dict.pop(prefix + 'num_batches_tracked', None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight.reshape(1, -1, 1, 1) * \
                (self.running_var.reshape(1, -1, 1, 1) + self.eps).rsqrt()
        bias = self.bias.reshape(1, -1, 1, 1) - \
               self.running_mean.reshape(1, -1, 1, 1) * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """
    Base wrapper for backbone networks with layer selection and mask resizing.
    """
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int):
        super().__init__()
        for name, param in backbone.named_parameters():
            if not train_backbone or all(x not in name for x in ['layer2', 'layer3', 'layer4']):
                param.requires_grad_(False)

        self.body = IntermediateLayerGetter(backbone, return_layers={'layer4': '0'})
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        features = self.body(tensor_list.tensors)
        out = {}
        mask = tensor_list.mask

        for name, x in features.items():
            assert mask is not None
            resized_mask = F.interpolate(mask[None].float(), size=x.shape[-2:], mode='nearest')[0].to(torch.bool)
            out[name] = NestedTensor(x, resized_mask)

        return out


class Backbone(BackboneBase):
    """
    ResNet backbone with frozen BatchNorm.
    """
    def __init__(self, name: str, train_backbone: bool, dilation: bool):
        assert name in torchvision.models.__dict__, f"Unknown backbone: {name}"
        backbone_fn = getattr(torchvision.models, name)

        backbone = backbone_fn(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d
        )
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels)


class Joiner(nn.Sequential):
    """
    Combines the backbone and positional encoding into a single module.
    """
    def __init__(self, backbone: nn.Module, position_encoding: nn.Module):
        super().__init__(backbone, position_encoding)

    def forward(self, tensor_list: NestedTensor):
        features = self[0](tensor_list)
        out, pos = [], []

        for x in features.values():
            out.append(x)
            pos_encoding = self[1](x).to(dtype=x.tensors.dtype)
            pos.append(pos_encoding)

        return out, pos


def build_backbone(args) -> nn.Module:
    """
    Builds the full backbone module including position encoding.
    """
    position_encoding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.backbone, train_backbone, args.dilation)
    model = Joiner(backbone, position_encoding)
    model.num_channels = backbone.num_channels
    return model

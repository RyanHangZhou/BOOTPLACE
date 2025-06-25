import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a standard sinusoidal position embedding module (as in DETR).

    It encodes the spatial position (x, y) using sine and cosine functions at different frequencies.
    This version supports normalized coordinates if `normalize=True`, as used in COCO-style images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        Args:
            num_pos_feats: number of positional features per spatial axis (half of the final dim)
            temperature: frequency scale for sin/cos
            normalize: whether to normalize x/y before encoding
            scale: scale factor for normalization (default: 2π)
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize must be True if scale is provided")

        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: a NestedTensor with 'tensors' of shape [B, C, H, W]
                         and 'mask' of shape [B, H, W] indicating padded regions.
        Returns:
            pos: position encoding of shape [B, 2*num_pos_feats, H, W]
        """
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=4).flatten(3)

        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute position embedding with learnable parameters.
    This is similar to DETR's learned position encoding.
    """

    def __init__(self, num_pos_feats=256, max_h=50, max_w=50):
        """
        Args:
            num_pos_feats: number of positional features per axis (final output dim is 2*num_pos_feats)
            max_h: maximum height to support (used for row embedding)
            max_w: maximum width to support (used for column embedding)
        """
        super().__init__()
        self.row_embed = nn.Embedding(max_h, num_pos_feats)
        self.col_embed = nn.Embedding(max_w, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor with .tensors of shape [B, C, H, W]
        Returns:
            pos: learned position embedding of shape [B, 2*num_pos_feats, H, W]
        """
        x = tensor_list.tensors
        h, w = x.shape[-2:]

        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),    # shape: [H, W, C]
            y_emb.unsqueeze(1).repeat(1, w, 1),    # shape: [H, W, C]
        ], dim=-1)

        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)  # [B, 2C, H, W]
        return pos


def build_position_encoding(args):
    """
    Factory to construct a position encoding module based on string config.
    Args:
        args.position_embedding: one of ['sine', 'v2', 'learned', 'v3']
        args.hidden_dim: model's hidden dimension (e.g. 256)
    """
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"Unsupported position embedding type: {args.position_embedding}")
    return position_embedding

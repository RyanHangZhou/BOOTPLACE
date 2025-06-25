"""
Detection Transformer class.

Modified from torch.nn.Transformer with the following changes:
- Positional encodings are passed into Multi-head Attention explicitly
- Extra LayerNorm at the end of the encoder is removed
- Decoder returns activations from all decoding layers
- Additional patch feature encoder/decoder is included for patch-level CLIP-style alignment
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# ---- Main Transformer Module ----

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 normalize_before=False, return_intermediate_dec=False,
                 is_mask=False):
        """
        Args:
            d_model: feature dimension (default: 512)
            nhead: number of attention heads
            num_encoder_layers: number of layers in encoder
            num_decoder_layers: number of layers in decoder
            dim_feedforward: hidden dimension of feedforward layers
            dropout: dropout probability
            activation: activation function in FFN (default: relu)
            normalize_before: whether to apply layer norm before attention
            return_intermediate_dec: whether to return all decoder layers’ outputs
            is_mask: custom logic for masked decoding
        """
        super().__init__()

        # Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        # Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, normalize_before, is_mask)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec)

        # Patch feature encoder and decoder
        self.patch_encoder = PatchEncoder(out_dim=d_model)
        self.patch_decoder = PatchDecoder()  # placeholder

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, patches, query_embed, pos_embed):
        """
        Args:
            src: [B, C, H, W] - input image features
            mask: [B, H, W] - padding mask
            patches: [B, C_patch, H_patch_total, patch_size] - CLIP-style region patches
            query_embed: [num_queries, C] - learnable object queries
            pos_embed: [B, C, H, W] - positional encoding
        Returns:
            hs: [layers, B, num_queries, d_model] - decoder outputs
            memory: [B, C, H, W] - encoder output features
            out_patch_feat: [B, num_patches, d_model] - encoded patch features
        """
        bs, c, h, w = src.shape

        # Flatten input
        src = src.flatten(2).permute(2, 0, 1)            # [HW, B, C]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)                           # [B, HW]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)              # [num_queries, B, C]

        # Patch encoder
        _, patch_c, patch_h_total, patch_s = patches.shape
        n_patch = patch_h_total // patch_s
        device = patches.device
        out_patch_feat = torch.zeros((bs, n_patch, self.d_model), device=device)

        for i in range(n_patch):
            patch = patches[:, :, i * patch_s : (i + 1) * patch_s, :]  # [B, C, patch_s, patch_size]
            feat = self.patch_encoder(patch)                           # [B, d_model]
            out_patch_feat[:, i, :] = feat

        # Transformer encoder
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # [HW, B, C]

        # Transformer decoder
        hs = self.decoder(
            tgt, memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed)

        # Reshape memory back to image space
        memory_img = memory.permute(1, 2, 0).view(bs, c, h, w)

        return hs.transpose(1, 2), memory_img, out_patch_feat



class PatchEncoder(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 16 * 16, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)


class PatchDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        return self.relu(self.deconv2(x))


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def forward_post(self, src, src_mask, src_key_padding_mask, pos):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))

    def forward_pre(self, src, src_mask, src_key_padding_mask, pos):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(self.norm2(src)))))
        return src + self.dropout2(src2)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, is_mask=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.is_mask = is_mask

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask,
                                    pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)

    def forward_post(self, tgt, memory,
                     tgt_mask, memory_mask,
                     tgt_key_padding_mask, memory_key_padding_mask,
                     pos, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        if self.is_mask:
            L_q, L_m = tgt.size(0), memory.size(0)
            memory_mask = torch.triu(torch.ones(L_q, L_m, device=tgt.device) * float('-inf'), diagonal=1)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               self.with_pos_embed(memory, pos),
                               memory,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))

    def forward_pre(self, tgt, memory,
                    tgt_mask, memory_mask,
                    tgt_key_padding_mask, memory_key_padding_mask,
                    pos, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt2, query_pos),
                               self.with_pos_embed(memory, pos),
                               memory,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(tgt2))))
        return tgt + self.dropout3(tgt2)


def _get_clones(module, N):
    """
    Return N deep-copied modules in a ModuleList.
    Used for stacking encoder/decoder layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """
    Return an activation function given a string.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Unsupported activation function: {activation}")


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        is_mask=args.is_mask
    )
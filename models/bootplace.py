import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import List, Dict, Union

from util import box_ops
from util.misc import (
    NestedTensor, nested_tensor_from_tensor_list,
    accuracy, get_world_size, interpolate, is_dist_avail_and_initialized
)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer


class BOOTPLACE(nn.Module):
    """
    BOOTPLACE: Transformer-based model for multi-object placement detection.
    """
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        num_classes: int,
        num_queries: int,
        aux_loss: bool = False
    ):
        """
        Args:
            backbone (nn.Module): Backbone network to extract image features.
            transformer (nn.Module): Transformer for object reasoning and matching.
            num_classes (int): Number of target object classes.
            num_queries (int): Maximum number of object queries per image.
            aux_loss (bool): Whether to include auxiliary losses from intermediate decoder layers.
        """
        super().__init__()
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.backbone = backbone
        self.transformer = transformer

        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = cMLP(hidden_dim, hidden_dim, 4, 3, num_queries)

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(
        self,
        samples: Union[NestedTensor, List[torch.Tensor], torch.Tensor],
        patches: Union[NestedTensor, List[torch.Tensor], torch.Tensor],
        targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            samples (NestedTensor or list): Input scene image tensor(s).
            patches (NestedTensor or list): Object patch tensor(s).
            targets (list of dict): Each dict contains:
                - 'boxes': Tensor [N_i, 4] of GT box coordinates (normalized xywh).
                - 'negative_bin_mask': Tensor [N_i] indicating object validity.

        Returns:
            dict: Output predictions containing:
                - pred_logits: [B, num_queries, num_classes+1]
                - pred_boxes:  [B, num_queries, 4]
                - pred_clip:   similarity scores between image and patch features
                - pred_clip2:  transpose of pred_clip
                - aux_outputs: (optional) intermediate layer outputs
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if isinstance(patches, (list, torch.Tensor)):
            patches = nested_tensor_from_tensor_list(patches)

        patches = patches.decompose()[0]
        device = patches.device
        B = len(targets)

        # Initialize target tensor placeholders
        # target_boxes = torch.zeros(B, self.num_queries, 4, device=device)
        # target_mask = torch.zeros(B, self.num_queries, device=device)

        # for i, tgt in enumerate(targets):
        #     num_obj = min(len(tgt['boxes']), len(tgt['negative_bin_mask']))
        #     target_boxes[i, :num_obj] = tgt['boxes'][:num_obj]
        #     target_mask[i, :num_obj] = tgt['negative_bin_mask'][:num_obj]

        # # Mask unused slots
        # target_mask = target_mask.unsqueeze(-1).expand_as(target_boxes)
        # scene_boxes = target_boxes * target_mask

        scene_boxes = torch.zeros(B, self.num_queries, 4, device=device)
        for i, tgt in enumerate(targets):
            num_obj = len(tgt['scene_boxes'])
            scene_boxes[i, :num_obj] = tgt['scene_boxes']

        # Backbone + Transformer
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        hs, _, patch_feat = self.transformer(
            self.input_proj(src),
            mask,
            patches,
            self.query_embed.weight,
            pos[-1]
        )

        # Feature similarity via dot product
        sim_image_to_patch, sim_patch_to_image = self.clip(hs[-1], patch_feat)

        # Prediction heads
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs, scene_boxes).sigmoid()

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_clip': sim_image_to_patch,
            'pred_clip2': sim_patch_to_image,
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    def clip(
        self,
        img_feat: torch.Tensor,     # [B, num_queries, D]
        patch_feat: torch.Tensor    # [B, D, P]
    ) -> (torch.Tensor, torch.Tensor):
        """
        Computes cosine similarity between image features and patch features.
        Returns logits scaled by learned temperature.
        """
        img_feat = F.normalize(img_feat, p=2, dim=-1)
        patch_feat = F.normalize(patch_feat, p=2, dim=-2)

        logit_scale = self.logit_scale.exp()
        patch_feat = patch_feat.transpose(1, 2)  # [B, P, D]

        logits_per_image = logit_scale * torch.bmm(img_feat, patch_feat)       # [B, Q, P]
        logits_per_patch = logits_per_image.transpose(1, 2)                    # [B, P, Q]

        return logits_per_image, logits_per_patch

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class: torch.Tensor,
        outputs_coord: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Collect auxiliary outputs from intermediate decoder layers.
        """
        return [
            {'pred_logits': cls, 'pred_boxes': box}
            for cls, box in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    """
    Computes the loss for BOOTPLACE using Hungarian matching.
    Steps:
        1) Match model predictions with targets using the matcher.
        2) Compute classification, box, mask, and patch similarity losses.
    """
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: List[str]
    ):
        """
        Args:
            num_classes: Number of foreground object classes (excluding 'no object').
            matcher: Module computing Hungarian matching between predictions and targets.
            weight_dict: Dict of loss names and their weights.
            eos_coef: Weight for the 'no object' class in classification loss.
            losses: List of activated losses (e.g. ['labels', 'boxes', 'clip']).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """ Cross-entropy classification loss with no-object weighting. """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_clip(self, outputs, targets, indices, num_boxes, log=True):
        """ Symmetric similarity loss between image regions and object patches. """
        src_logits = outputs['pred_clip']
        src_logits2 = outputs['pred_clip2']

        # Negative log-similarity
        log_output = torch.log(1 - F.softmax(src_logits, dim=-2) + 1e-6)
        log_output2 = torch.log(1 - F.softmax(src_logits2, dim=-1) + 1e-6)

        loss_clip0 = -log_output.mean()
        loss_clip1 = -log_output2.mean()
        loss_sim = (loss_clip0 + loss_clip1) / 2

        return {'loss_clip': loss_sim, 'clip_error': loss_sim} if log else {'loss_clip': loss_sim}

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Absolute error in predicted vs actual object counts (for logging only). """
        pred_logits = outputs['pred_logits']
        tgt_lengths = torch.as_tensor([len(t["labels"]) for t in targets], device=pred_logits.device)
        card_pred = (pred_logits.argmax(-1) != self.num_classes).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """ L1 + GIoU loss on predicted vs ground-truth boxes. """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)
        ))

        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss: str, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'clip': self.loss_clip,
            'cardinality': self.loss_cardinality,
        }
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute all enabled losses given model outputs and targets.

        Args:
            outputs: Dict containing model predictions.
            targets: List of ground-truth dicts per image.

        Returns:
            Dict of computed loss terms.
        """
        outputs_no_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_no_aux, targets)

        # Normalize by average number of GT boxes across devices
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute main losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Compute auxiliary layer losses if present
        if 'aux_outputs' in outputs:
            for i, aux_out in enumerate(outputs['aux_outputs']):
                indices_aux = self.matcher(aux_out, targets)
                for loss in self.losses:
                    if loss in {'masks', 'clip'}:
                        continue
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_out, targets, indices_aux, num_boxes, **kwargs)
                    losses.update({f"{k}_{i}": v for k, v in l_dict.items()})
        return losses


class PostProcess(nn.Module):
    """
    This module converts model outputs into COCO-style format for evaluation or visualization.
    """
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        Args:
            outputs: dict containing 'pred_logits' and 'pred_boxes' (normalized cxcywh format)
            target_sizes: Tensor of shape [batch_size, 2], each row is (height, width)

        Returns:
            List[Dict]: each dict contains keys: 'scores', 'labels', 'boxes' in absolute xyxy format
        """
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        assert logits.shape[0] == target_sizes.shape[0]
        assert target_sizes.shape[1] == 2

        prob = F.softmax(logits, dim=-1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        scale_fct = torch.stack([target_sizes[:, 1], target_sizes[:, 0],
                                 target_sizes[:, 1], target_sizes[:, 0]], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]
        return results



class cMLP(nn.Module):
    """
    A location-conditioned multi-layer perceptron that modulates predictions based on target layout.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_queries):
        super().__init__()
        self.num_layers = num_layers
        layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i+1]) for i in range(num_layers)
        ])

        self.loc_proj = nn.Sequential(
            nn.Linear(num_queries * 4, num_queries),
            nn.ReLU(inplace=True),
            nn.Linear(num_queries, 1)
        )

    def forward(self, x, locations):
        loc_feat = self.loc_proj(locations.view(locations.size(0), -1)).view(-1, 1, 1)
        x = x * loc_feat
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class TransformerDecoderLayer(nn.Module):
    """
    A basic Transformer decoder layer with self-attention and feedforward sublayers.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2, _ = self.self_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2 = self.ffn(tgt)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        return tgt


class MaskedTransformerDecoder(nn.Module):
    """
    A stack of Transformer decoder layers with a final linear projection.
    """
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, bbox_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, bbox_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.output_proj(tgt)


def build(args):
    """
    Construct the BOOTPLACE model, criterion, and postprocessor.
    Args:
        args: argparse.Namespace with all necessary arguments
    Returns:
        model (nn.Module), criterion (nn.Module), postprocessors (dict)
    """
    device = torch.device(args.device)
    num_classes = 8

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = BOOTPLACE(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )

    matcher = build_matcher(args)

    weight_dict = {
        'loss_ce': args.ce_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_clip': args.clip_loss_coef,
    }

    if args.aux_loss:
        aux_weight_dict = {
            f"{k}_{i}": v
            for i in range(args.dec_layers - 1)
            for k, v in weight_dict.items()
        }
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=['labels', 'boxes', 'cardinality', 'clip']
    )
    criterion.to(device)

    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

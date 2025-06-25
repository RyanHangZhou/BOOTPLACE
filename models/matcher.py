import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between predictions and ground truth using the Hungarian algorithm.

    For efficiency, ground truth does not contain "no-object" entries, so the number of predictions is typically
    greater than the number of targets. The matcher finds a 1-to-1 best match per object instance.
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Args:
            cost_class: weight of the classification cost in total matching cost
            cost_bbox: weight of the L1 bounding box distance in total matching cost
            cost_giou: weight of the GIoU loss in total matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_patches = 100  # Assumes each batch has at most 100 candidate object patches

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, \
            "All cost weights cannot be 0."

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs matching between predicted boxes and ground truth boxes.

        Args:
            outputs: dict with keys:
                - "pred_logits": Tensor of shape [B, Q, C], classification logits
                - "pred_boxes": Tensor of shape [B, Q, 4], predicted boxes (cxcywh, normalized)

            targets: list[dict], each dict contains:
                - "labels": Tensor of shape [N_i], object classes
                - "boxes": Tensor of shape [N_i, 4], ground-truth boxes (cxcywh, normalized)

        Returns:
            List[Tuple[Tensor, Tensor]]: Each tuple contains (pred_indices, target_indices)
                  for one batch item, suitable for supervision.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten batch and queries for batched cost computation
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)   # [B*Q, C]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)                 # [B*Q, 4]

        # Concatenate targets across batch
        tgt_ids = torch.cat([t["labels"] for t in targets])           # [sum(N_i)]
        tgt_bbox = torch.cat([t["boxes"] for t in targets])           # [sum(N_i), 4]

        # Optional: map target IDs to global patch ID space (for clip loss)
        tgt_number = [len(t["labels"]) for t in targets]
        tgt_ids_clip = []
        for i in range(bs):
            start = i * self.num_patches
            tgt_ids_clip.extend(range(start, start + tgt_number[i]))

        # Compute classification cost: negative log-likelihood
        cost_class = -out_prob[:, tgt_ids]  # [B*Q, sum(N_i)]

        # Compute bbox L1 distance cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Combine weighted costs
        total_cost = self.cost_class * cost_class + \
                     self.cost_bbox * cost_bbox + \
                     self.cost_giou * cost_giou

        total_cost = total_cost.view(bs, num_queries, -1).cpu()

        sizes = [len(t["boxes"]) for t in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(total_cost.split(sizes, dim=-1))]

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


def build_matcher(args):
    """
    Construct HungarianMatcher from argparse config
    """
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
    )

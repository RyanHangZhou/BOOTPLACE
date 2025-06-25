from typing import Tuple

import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (center_x, center_y, width, height) format
    to (x_min, y_min, x_max, y_max) format.

    Args:
        x (torch.Tensor): Tensor of shape (..., 4)

    Returns:
        torch.Tensor: Converted boxes with same shape as input.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        x_c - 0.5 * w,
        y_c - 0.5 * h,
        x_c + 0.5 * w,
        y_c + 0.5 * h,
    ]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x_min, y_min, x_max, y_max) format
    to (center_x, center_y, width, height) format.

    Args:
        x (torch.Tensor): Tensor of shape (..., 4)

    Returns:
        torch.Tensor: Converted boxes with same shape as input.
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        x1 - x0,
        y1 - y0,
    ]
    return torch.stack(b, dim=-1)


def box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise Intersection-over-Union (IoU) between two sets of boxes,
    also returns the union area.

    Args:
        boxes1 (torch.Tensor): shape [N, 4] in xyxy format.
        boxes2 (torch.Tensor): shape [M, 4] in xyxy format.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - iou (torch.Tensor): [N, M] pairwise IoU matrix.
            - union (torch.Tensor): [N, M] pairwise union area matrix.
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter  # [N, M]

    iou = inter / union
    return iou, union


def generalized_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Compute Generalized Intersection over Union (GIoU) between two sets of boxes.

    The boxes must be in (x_min, y_min, x_max, y_max) format.

    Args:
        boxes1 (torch.Tensor): shape [N, 4]
        boxes2 (torch.Tensor): shape [M, 4]

    Returns:
        torch.Tensor: [N, M] pairwise GIoU matrix.
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 are not valid"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 are not valid"

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]  # enclosing box area

    return iou - (area - union) / area


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute bounding boxes in xyxy format around the provided masks.

    Args:
        masks (torch.Tensor): Binary masks of shape [N, H, W].

    Returns:
        torch.Tensor: Bounding boxes [N, 4] in (x_min, y_min, x_max, y_max) format.
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float32, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float32, device=masks.device)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    x_mask = masks * x_grid.unsqueeze(0)  # [N, H, W]
    x_max = x_mask.flatten(1).max(dim=1)[0]
    x_min = x_mask.masked_fill(~masks.bool(), 1e8).flatten(1).min(dim=1)[0]

    y_mask = masks * y_grid.unsqueeze(0)
    y_max = y_mask.flatten(1).max(dim=1)[0]
    y_min = y_mask.masked_fill(~masks.bool(), 1e8).flatten(1).min(dim=1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

import math
import sys
import os
from typing import Iterable, Optional, Tuple, Dict, Any

import torch
from torch import nn
from torch.optim import Optimizer

import util.misc as utils
from pycoco.coco_eval import CocoEvaluator
from pycocotools.coco import COCO


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
) -> Dict[str, float]:
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss criterion.
        data_loader (Iterable): Data loader yielding (samples, patches, targets).
        optimizer (Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device on which to perform computation.
        epoch (int): Current epoch number (for logging).
        max_norm (float, optional): Max norm for gradient clipping. Default: 0.0 (disabled).

    Returns:
        Dict[str, float]: Dictionary of averaged training metrics.
    """
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = f'Epoch: [{epoch}]'
    print_freq = 10

    for samples, patches, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        patches = patches.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, patches, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        # Reduce losses across all GPUs for logging consistency
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # Update metric logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced.get('class_error', 0.0))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Synchronize metrics between processes (if distributed)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def ensure_dir(dir_path: str) -> None:
    """
    Ensure that a directory exists; create it if it does not.

    Args:
        dir_path (str): Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    postprocessors: dict,
    data_loader: torch.utils.data.DataLoader,
    base_ds: COCO,
    device: torch.device,
    output_dir: str,
    data_path: str,
) -> Tuple[Dict[str, float], Optional[CocoEvaluator]]:
    """
    Evaluate the model on the validation/test dataset.

    Args:
        model (nn.Module): The trained model.
        criterion (nn.Module): The loss criterion used during evaluation.
        postprocessors (dict): Dictionary of post-processing functions for outputs,
                               e.g., {'bbox': bbox_postprocessor, 'segm': segm_postprocessor}.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        base_ds (COCO): COCO object of the base dataset for evaluation.
        device (torch.device): Device to run evaluation on.
        output_dir (str): Directory to save outputs if needed.
        data_path (str): Root path to dataset (for loading annotations, etc.).

    Returns:
        stats (dict): Dictionary of averaged evaluation metrics.
        coco_evaluator (CocoEvaluator or None): The COCO evaluator object after evaluation.
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors)
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # Load COCO annotations for test set (optional, for reference)
    _ = COCO(os.path.join(data_path, 'test', 'annotations.json'))

    for samples, patches, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        patches = patches.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, patches, targets)

        # Compute losses
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # Reduce losses across GPUs if distributed
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}

        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced.get('class_error', 0.0))

        # Prepare original target sizes for post-processing
        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)  # e.g., [batch_size, 2]
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if 'segm' in postprocessors:
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        # Map target ids to outputs for COCO evaluation
        res = {target['ids'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # Synchronize metrics between processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # Add COCO evaluation stats if available
    if coco_evaluator is not None:
        if 'bbox' in postprocessors:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
    
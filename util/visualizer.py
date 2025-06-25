import os
import datetime
from typing import Optional, Dict, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
from textwrap import wrap

from pycocotools import mask as maskUtils


def renorm(img: torch.Tensor, mean=None, std=None) -> torch.Tensor:
    """
    Denormalize image tensor (3,H,W) or (B,3,H,W) to [0,1] RGB.

    Args:
        img (Tensor): Normalized tensor image
        mean (list): Mean used for normalization
        std (list): Std used for normalization

    Returns:
        Tensor: Denormalized image
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    assert img.dim() in (3, 4), f"Expected img of dim 3 or 4, got {img.dim()}"
    assert img.size(-3) == 3, f"Expected 3 channels, got {img.size(-3)}"

    mean = torch.tensor(mean, device=img.device)
    std = torch.tensor(std, device=img.device)

    if img.dim() == 3:
        return (img * std[:, None, None] + mean[:, None, None])
    else:
        return (img * std[None, :, None, None] + mean[None, :, None, None])


class ColorMap:
    def __init__(self, base_rgb=(255, 255, 0)):
        self.base_rgb = np.array(base_rgb)

    def __call__(self, attn_map: np.ndarray) -> np.ndarray:
        """
        Convert 1-channel attention map to RGBA using base color.

        Args:
            attn_map (np.ndarray): Attention map (H, W), uint8

        Returns:
            np.ndarray: RGBA image (H, W, 4)
        """
        assert attn_map.dtype == np.uint8
        h, w = attn_map.shape
        rgb = np.tile(self.base_rgb[None, None, :], (h, w, 1))
        rgba = np.concatenate((rgb, attn_map[..., None]), axis=-1)
        return rgba.astype(np.uint8)


class COCOVisualizer:
    """
    Visualize COCO-format bounding box annotations.
    """

    def __init__(self):
        pass

    def visualize(
        self,
        img: torch.Tensor,
        tgt: Dict,
        caption: Optional[str] = None,
        dpi: int = 120,
        savedir: Optional[str] = None,
        show_in_console: bool = True
    ) -> None:
        """
        Visualize and optionally save a bounding box image.

        Args:
            img (Tensor): Image tensor (3,H,W)
            tgt (Dict): Target dict containing at least 'boxes', 'size'
            caption (str): Optional string label
            dpi (int): Resolution
            savedir (str): Save path
            show_in_console (bool): Whether to call plt.show()
        """
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = 5
        ax = plt.gca()
        img = renorm(img).permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)

        self._add_target_boxes(tgt)

        if savedir:
            filename = f"{caption or 'vis'}-{int(tgt['image_id'])}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
            full_path = os.path.join(savedir, filename)
            os.makedirs(savedir, exist_ok=True)
            plt.savefig(full_path)
            print(f"[Visualizer] Saved to {full_path}")

        if show_in_console:
            plt.show()
        plt.close()

    def savefig(self, img: torch.Tensor, tgt: Dict, filename: str, savedir: str, dpi: int = 120) -> None:
        """
        Save visualization image with bounding boxes.

        Args:
            img (Tensor): Image tensor
            tgt (Dict): Target dict
            filename (str): Output filename (no extension)
            savedir (str): Output directory
        """
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = 5
        ax = plt.gca()
        img = renorm(img).permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)

        self._add_target_boxes(tgt)

        path = os.path.join(savedir, f"{filename}_box.png")
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(path)
        plt.close()

    def _add_target_boxes(self, tgt: Dict[str, Union[torch.Tensor, list]]) -> None:
        """
        Add bounding boxes and labels to the plot.
        """
        assert 'boxes' in tgt and 'size' in tgt, "Target must include 'boxes' and 'size'"
        ax = plt.gca()
        H, W = tgt['size']
        boxes = tgt['boxes'].cpu()
        labels = tgt.get('box_label', [])
        caption = tgt.get('caption', None)

        polygons, colors, label_positions = [], [], []

        for i, box in enumerate(boxes):
            cx, cy, w, h = (box * torch.tensor([W, H, W, H])).tolist()
            x0, y0 = cx - w / 2, cy - h / 2
            poly = np.array([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h], [x0 + w, y0]])
            polygons.append(Polygon(poly))
            c = (np.random.rand(3) * 0.6 + 0.4).tolist()
            colors.append(c)
            label_positions.append((x0, y0))

        patch_fill = PatchCollection(polygons, facecolor=colors, linewidths=0, alpha=0.1)
        patch_edge = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=2)
        ax.add_collection(patch_fill)
        ax.add_collection(patch_edge)

        for i, pos in enumerate(label_positions):
            if labels:
                ax.text(pos[0], pos[1], str(labels[i]), color='black',
                        bbox={'facecolor': colors[i], 'alpha': 0.6, 'pad': 1})

        if caption:
            ax.set_title("\n".join(wrap(caption, 60)))

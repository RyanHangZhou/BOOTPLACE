# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Data augmentation transforms for images and bounding boxes.
"""
import random
from typing import Tuple, Optional, List

import PIL.Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh


# --- Helper Functions --- #
def crop(image, patches, target, region):
    cropped_image = F.crop(image, *region)
    target = target.copy()

    i, j, h, w = region
    target["size"] = torch.tensor([h, w])

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.tensor([j, i, j, i], dtype=torch.float32)
        cropped_boxes = torch.min(cropped_boxes.view(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1] - cropped_boxes[:, 0]).prod(dim=1)

        keep = (cropped_boxes[:, 1] > cropped_boxes[:, 0]).all(dim=1)
        target["boxes"] = cropped_boxes.view(-1, 4)[keep]
        target["area"] = area[keep]
        target["labels"] = target["labels"][keep]

    return cropped_image, patches, target


def hflip(image, patches, target):
    flipped_image = F.hflip(image)
    flipped_patches = F.hflip(patches)
    w, _ = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1], dtype=torch.float32) + torch.tensor([w, 0, w, 0])
        target["boxes"] = boxes

    return flipped_image, flipped_patches, target


def resize(image, patches, target, size, max_size=None):
    def get_size(img_size):
        w, h = img_size
        if isinstance(size, (list, tuple)):
            return size[::-1]
        adjusted_size = size
        if max_size is not None:
            min_original, max_original = min(w, h), max(w, h)
            if max_original / min_original * size > max_size:
                adjusted_size = int(round(max_size * min_original / max_original))
        if w < h:
            return int(adjusted_size * h / w), adjusted_size
        return adjusted_size, int(adjusted_size * w / h)

    new_h, new_w = get_size(image.size)
    resized_image = F.resize(image, (new_h, new_w))

    target = target.copy() if target else None
    if target and "boxes" in target:
        ratio_w, ratio_h = new_w / image.width, new_h / image.height
        scale = torch.tensor([ratio_w, ratio_h, ratio_w, ratio_h], dtype=torch.float32)
        target["boxes"] = target["boxes"] * scale
        target["area"] = target["area"] * (ratio_w * ratio_h)
        target["size"] = torch.tensor([new_h, new_w])

    return resized_image, patches, target


def pad(image, patches, target, padding: Tuple[int, int]):
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target:
        target = target.copy()
        target["size"] = torch.tensor(padded_image.size[::-1])
    return padded_image, patches, target


# --- Transform Classes --- #
class RandomCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img, patches, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, patches, target, region)


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, patches, target):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, patches, target, region)


class CenterCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img, patches, target):
        iw, ih = img.size
        ch, cw = self.size
        top = (ih - ch) // 2
        left = (iw - cw) // 2
        return crop(img, patches, target, (top, left, ch, cw))


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, patches, target):
        if random.random() < self.p:
            return hflip(img, patches, target)
        return img, patches, target


class RandomResize:
    def __init__(self, sizes: List[int], max_size: Optional[int] = None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, patches, target=None):
        size = random.choice(self.sizes)
        return resize(img, patches, target, size, self.max_size)


class RandomPad:
    def __init__(self, max_pad: int):
        self.max_pad = max_pad

    def __call__(self, img, patches, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, patches, target, (pad_x, pad_y))


class RandomSelect:
    def __init__(self, transform1, transform2, p=0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.p = p

    def __call__(self, img, patches, target):
        if random.random() < self.p:
            return self.transform1(img, patches, target)
        return self.transform2(img, patches, target)


class ToTensor:
    def __call__(self, img, patches, target):
        return F.to_tensor(img), F.to_tensor(patches), target


class RandomErasing:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, patches, target):
        return self.eraser(img), patches, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, patches, target=None):
        img = F.normalize(img, mean=self.mean[:3], std=self.std[:3])
        patches = F.normalize(patches, mean=self.mean, std=self.std)

        if target:
            target = target.copy()
            h, w = img.shape[-2:]
            if "boxes" in target:
                boxes = box_xyxy_to_cxcywh(target["boxes"])
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes

        return img, patches, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, patches, target):
        for t in self.transforms:
            img, patches, target = t(img, patches, target)
        return img, patches, target

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(str(t) for t in self.transforms)})"

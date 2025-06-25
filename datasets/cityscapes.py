import os
import re
import random
import numpy as np
import imageio
from pathlib import Path
from PIL import Image
from skimage.transform import resize, AffineTransform, warp
from pycocotools.coco import COCO as AnnLoader

import torch
import torch.utils.data

import datasets.transforms as T
from util.box_ops import box_xyxy_to_cxcywh


class Cityscapes(torch.utils.data.Dataset):
    def __init__(self, img_folder, object_folder, ann_file, split, transforms):
        self.img_folder = img_folder
        self.object_folder = object_folder
        self.ann_data = AnnLoader(ann_file)
        self.ids = sorted(self.ann_data.imgs.keys())
        self.transforms = transforms
        self.label_packer = PackLabels()
        self.split = split

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ann_data.loadImgs(self.ids[idx])[0]["file_name"].rsplit(".", 1)[0]

        image = self._load_image(image_id)
        object_patches = self._load_objects(image_id)
        annotations = self._load_target(self.ids[idx], self.split)

        target = {'image_id': image_id, 'annotations': annotations}
        image, target = self.label_packer(image, object_patches, target, self.ids[idx], self.split)

        if self.transforms is not None:
            image, object_patches, target = self.transforms(image, object_patches, target)

        return image, object_patches, target

    def _load_image(self, image_id):
        image = Image.open(os.path.join(self.img_folder, f"{image_id}.png")).convert("RGB")
        return image.resize((960, 540))

    def _load_objects(self, image_id):
        max_objects = 100
        patch_size = 64
        class_list = ['car', 'person', 'rider', 'train', 'bus', 'bicycle', 'truck', 'motorcycle']

        object_dir = os.path.join(self.object_folder, image_id)
        if not os.path.isdir(object_dir):
            patch = imageio.imread(object_dir+'.png')
            patch = resize(patch, (patch_size, patch_size))
            return Image.fromarray((patch * 255).astype(np.uint8))

        object_list = self._sort_filenames_by_number(os.listdir(object_dir))[:max_objects]
        object_canvas = np.zeros((max_objects * patch_size, patch_size, 4))

        for idx, filename in enumerate(object_list):
            class_idx = class_list.index(filename[:-7])
            patch = imageio.imread(os.path.join(object_dir, filename))
            patch = resize(patch, (patch_size, patch_size))
            object_canvas[idx * patch_size:(idx + 1) * patch_size] = patch

        return Image.fromarray((object_canvas * 255).astype(np.uint8))

    def _load_target(self, image_id, split='train'):
        if split == 'train':
            ann_ids = self.ann_data.getAnnIds(image_id)
            return self.ann_data.loadAnns(ann_ids)

        image_info = self.ann_data.loadImgs(image_id)[0]
        image_name = image_info["file_name"].rsplit(".", 1)[0]

        gt_location_file = self.img_folder.parent / 'location_single' / f'{image_name}.txt'
        scene_location_dir = self.img_folder.parent / 'location' / image_name

        h, w = 540, 960

        # Load target bounding box (single box for this image)
        try:
            with open(gt_location_file, "r") as f:
                values = list(map(float, f.readlines()))
            assert len(values) == 4, f"Expected 4 values in {gt_location_file}, got {len(values)}"
        except Exception as e:
            raise RuntimeError(f"Error loading target box from {gt_location_file}: {e}")

        target_box = torch.tensor([values], dtype=torch.float32)

        # Load all scene boxes (excluding the target box itself)
        scene_boxes = []
        if os.path.isdir(scene_location_dir):
            for fname in os.listdir(scene_location_dir):
                file_path = scene_location_dir / fname
                try:
                    with open(file_path, "r") as f:
                        box_vals = list(map(float, f.readlines()))
                    if len(box_vals) != 4:
                        continue
                    if not np.allclose(box_vals, values):
                        scene_boxes.append(box_vals)
                except Exception:
                    continue
        else:
            raise FileNotFoundError(f"Scene directory not found: {scene_location_dir}")

        if len(scene_boxes) == 0:
            scene_boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
        else:
            scene_boxes_tensor = torch.tensor(scene_boxes, dtype=torch.float32)

        return {
            'boxes': target_box,          # (1, 4)
            'scene_boxes': scene_boxes_tensor  # (N, 4)
        }

    @staticmethod
    def _sort_filenames_by_number(filenames):
        def extract_numbers(text):
            return [float(num) for num in re.findall(r'\d+(?:\.\d+)?', text)]
        return sorted(filenames, key=extract_numbers)


class PackLabels:
    def __call__(self, image, object_patches, target, image_id, split):
        w, h = image.size
        annos = target["annotations"]

        if split == 'train':
            boxes = torch.as_tensor([obj["bbox"] for obj in annos], dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

            classes = torch.tensor([obj["category_id"] for obj in annos], dtype=torch.int64)
            area = torch.tensor([obj["area"] for obj in annos])

            valid = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes, classes, area = boxes[valid], classes[valid], area[valid]

            k = random.randint(1, len(classes))
            keep_pos = random.sample(range(len(classes)), k)
            keep_neg = [i for i in range(len(classes)) if i not in keep_pos]

            image = np.array(image).astype(int)
            object_patches = np.array(object_patches).astype(int)
            image = self._compose_patches(image, object_patches, boxes, keep_neg)

            mask_neg = np.zeros(len(classes), dtype=bool)
            mask_neg[keep_neg] = True

            scene_boxes = boxes[mask_neg]
            boxes, classes, area = boxes[~mask_neg], classes[~mask_neg], area[~mask_neg]
            image = Image.fromarray(np.uint8(image))
            packed = {
                "boxes": boxes,
                "labels": classes,
                "area": area,
                "ids": torch.tensor([image_id]),
                "size": torch.tensor([h, w]),
                "scene_boxes": scene_boxes
            }
        else:
            image = Image.fromarray(np.uint8(np.array(image)))

            packed = {
                "boxes": annos["boxes"],
                "ids": torch.tensor([image_id]),
                "size": torch.tensor([h, w]),
                "area": torch.tensor([1]),
                "labels": torch.tensor(list(range(len(annos["boxes"])))),
                "scene_boxes": annos["scene_boxes"],
            }

        return image, packed

    def _compose_patches(self, image, patches, boxes, indices):
        scale = patches.shape[1]
        for i in indices:
            patch = patches[i * scale:(i + 1) * scale]
            box = box_xyxy_to_cxcywh(boxes[i])
            image = self._compose_patch(image, patch, box)
        return image

    def _compose_patch(self, image, patch, box):
        h, w = image.shape[:2]
        mask = patch[..., 3:] / 255.0
        obj = patch[..., :3] * mask

        cx, cy, bw, bh = box.cpu().numpy()
        asp = self._aspect_ratio(patch)
        scale = bw if asp < bw / bh else bh

        canvas, mask_canvas = np.zeros((h, w, 3)), np.zeros((h, w, 1))
        canvas[:patch.shape[0], :patch.shape[1]] = obj
        mask_canvas[:patch.shape[0], :patch.shape[1]] = mask

        tform = AffineTransform(scale=scale / patch.shape[0], translation=(cx - scale / 2, cy - scale / 2))
        obj_warped = warp(canvas, tform.inverse)
        mask_warped = warp(mask_canvas, tform.inverse)

        composed = image * (1 - mask_warped) + obj_warped * mask_warped
        return np.clip(composed, 0, 255).astype(int)

    @staticmethod
    def _aspect_ratio(patch):
        mask = patch[..., 0] > 0
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        h_indices = np.where(rows)[0]
        w_indices = np.where(cols)[0]
        if len(h_indices) == 0 or len(w_indices) == 0:
            return 1.0
        return (w_indices[-1] - w_indices[0] + 1) / (h_indices[-1] - h_indices[0] + 1)


def make_cityscapes_transforms(split):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.440, 0.440, 0.440, 0], [0.320, 0.319, 0.318, 0.5])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if split == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    elif split == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=960),
            normalize,
        ])

    else:
        raise ValueError(f"Unknown split: {split}")


def build(split, args):
    root = Path(args.data_path)
    assert root.exists(), f"Provided data path {root} does not exist"

    paths = {
        'train': (root / 'train' / 'backgrounds', root / 'train' / 'objects', root / 'train' / 'annotations.json'),
        'test': (root / 'test' / 'backgrounds_single', root / 'test' / 'objects_single', root / 'test' / 'annotations.json')
    }

    img_folder, object_folder, ann_file = paths[split]
    dataset = Cityscapes(img_folder, object_folder, ann_file, split, transforms=make_cityscapes_transforms(split))
    return dataset

import datasets.transforms as T
import imageio
import numpy as np
import os
import random
import re
import torch
import torch.utils.data

from pathlib import Path
from PIL import Image, ImageFilter
from pycocotools.coco import COCO as AnnLoader
from skimage.transform import resize, AffineTransform, warp
from util.box_ops import box_xyxy_to_cxcywh


class Cityscapes(torch.utils.data.Dataset):
    def __init__(self, img_folder, object_folder, ann_file, split, transforms):
        self.img_folder = img_folder
        self.object_folder = object_folder
        self.ann_data = AnnLoader(ann_file)
        self.ids = list(sorted(self.ann_data.imgs.keys()))
        self.transforms = transforms
        self.pack_label = PackLabels()
        self.split = split
    

    def __len__(self) -> int:
        return len(self.ids)


    def __getitem__(self, idx):
        ids = self.ids[idx]
        image_id = self.ann_data.loadImgs(ids)[0]["file_name"][:-4]

        image = self._load_image(image_id) # [540, 960, 3]
        object_patches = self._load_objects(image_id) # [6400, 64, 4]
        target = self._load_target(ids) # [10]

        target = {'image_id': image_id, 'annotations': target}
        image, target, object_name = self.pack_label(image, object_patches, target, ids, self.split)

        if self.transforms is not None:
            image, object_patches, target = self.transforms(image, object_patches, target)

        return image, object_patches, target

    
    def _load_image(self, ids):
        return Image.open(os.path.join(self.img_folder, ids + '.png')).convert("RGB")
    

    def _load_objects(self, ids):
        object_dir = os.path.join(self.object_folder, ids)

        object_num = 100
        object_scale = 64
        class_list = ['car', 'person', 'rider', 'train', 'bus', 'bicycle', 'truck', 'motorcycle']

        object_list = os.listdir(object_dir)
        object_list = self.sort_list_by_number_only(object_list)[:object_num]
        counter = 0
        object_canvas = np.zeros((object_num*object_scale, object_scale, 4))
        object_classes = np.zeros((object_num), np.int)
        for location_file in object_list:
            object_classes[counter] = class_list.index(location_file[:-7])
            patch_sample_path = os.path.join(object_dir, location_file)
            patch_sample = imageio.imread(patch_sample_path)
            patch_sample = resize(patch_sample, (object_scale, object_scale))
            object_canvas[counter*object_scale:(counter+1)*object_scale, :] = patch_sample
            counter += 1
        object_canvas = Image.fromarray(np.uint8(object_canvas*255))

        return object_canvas
    

    def _load_target(self, ids):
        return self.ann_data.loadAnns(self.ann_data.getAnnIds(ids))
    

    def sort_list_by_number_only(self, lst):
        def extract_numbers(sample):
            return [float(s) for s in re.findall(r'\d+(?:\.\d+)?', sample)]
        return sorted(lst, key=extract_numbers)


class PackLabels(object):
    def __call__(self, image, object_patches, target, ids, split):
        class_num = 8
        w, h = image.size
        anno = target["annotations"]

        ''' 1. boxes '''
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        ''' 2. classes '''
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        ''' 3. area '''
        area = torch.tensor([obj["area"] for obj in anno])

        ''' 4. sample ID in jason format '''
        ids = torch.tensor([ids])

        ''' 5. object_patch name '''
        object_name = [obj["object_name"] for obj in anno]

        ''' 6. filter based on boxes '''
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        # object_name = object_name[keep] # should be filtered, but it's good without it
        object_name = [obj for i, obj in enumerate(object_name) if keep[i]]
        area = area[keep]

        ''' 7. prepare positive and negative objects supervision '''
        '''positive objects: for supervision; negative objects: for location encoder'''
        if split == 'train':
            bbx_num = len(classes)
            # random.seed(3)
            k = random.randint(1, bbx_num)  # Randomly choose k from 1 to N
            keep_positive = random.sample(range(bbx_num), k)
            keep_negative = [x for x in range(bbx_num) if x not in keep_positive]
            image = np.asarray(image).astype(int)
            object_patches = np.asarray(object_patches).astype(int)
            image = self.patches_compose(image, np.array(object_patches), boxes, keep_negative)

            keep_negative_bool = np.zeros_like(range(bbx_num), dtype=bool)
            keep_negative_bool[keep_negative] = 1
            
            classes = classes[~keep_negative_bool]
            boxes = boxes[~keep_negative_bool]
            area = area[~keep_negative_bool]

            image = Image.fromarray(np.uint8(image))
        else:
            bbx_num = len(classes)
            keep_negative = range(bbx_num)
            keep_positive = keep_negative # doesn't matter as it isn't used
            image = np.asarray(image).astype(int)
            image = Image.fromarray(np.uint8(image))

            keep_negative_bool = np.zeros_like(range(bbx_num), dtype=bool)
            keep_negative_bool[keep_negative] = 1


        ''' 7. pack them '''
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["area"] = area
        target["ids"] = ids
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["mask"] = torch.as_tensor(keep_positive)
        target["negative_bin_mask"] = torch.as_tensor(keep_negative_bool)

        return image, target, object_name
    
    def patches_compose(self, image, object_patches, boxes, object_list):
        scale = object_patches.shape[1]

        for i in object_list:
            patches_i = object_patches[i*scale:(i+1)*scale, :, :]
            boxes_i = box_xyxy_to_cxcywh(boxes[i, :])
            image = self.patch_compose(image, patches_i, boxes_i)
        
        return image
    
    def patch_compose(self, image, object_patch, bbx): 
        img_h, img_w = image.shape[:2]
        mask = object_patch[:, :, 3][:, :, None]
        obj = object_patch[:, :, :3]*(mask/255)
        p_s = object_patch.shape[0]

        bbx = bbx.cpu().detach().numpy()
        center_w = bbx[0]
        center_h = bbx[1]
        asp_ratio = self.get_aspect_ratio(object_patch)

        scale_w = bbx[2]
        scale_h = bbx[3]

        if asp_ratio < scale_w / scale_h:
            scale_ = scale_w
        else:
            scale_ = scale_h
        
        tmp_patch_canvas = np.zeros((img_h, img_w, 3))
        tmp_mask_canvas = np.zeros((img_h, img_w, 1))
        tmp_patch_canvas[:object_patch.shape[0], :object_patch.shape[1], :] = obj
        tmp_mask_canvas[:object_patch.shape[0], :object_patch.shape[1], :] = mask

        tform = AffineTransform(scale=scale_/p_s, translation=(center_w-scale_/2, center_h-scale_/2))
        tmp_patch_canvas = warp(tmp_patch_canvas, tform.inverse)
        tmp_mask_canvas = warp(tmp_mask_canvas, tform.inverse)
        tmp_mask_canvas = tmp_mask_canvas/255.

        image_out = image*(1-tmp_mask_canvas) + tmp_patch_canvas*tmp_mask_canvas
        
        return np.clip(image_out, 0, 255).astype(int)
    

    def get_aspect_ratio(self, object_patch):
        """ Get aspect ratio of an object_patch """

        img_h, img_w = object_patch.shape[:2]

        for i in range(img_h):
            line = object_patch[i, :, 0]
            if(np.sum(line) > 0):
                h_1 = i
                break
        for i in range(img_h-1, 0, -1):
            line = object_patch[i, :, 0]
            if(np.sum(line) > 0):
                h_2 = i
                break
        for i in range(img_w):
            column = object_patch[:, i, 0]
            if(np.sum(column) > 0):
                w_1 = i
                break
        for i in range(img_w-1, 0, -1):
            column = object_patch[:, i, 0]
            if(np.sum(column) > 0):
                w_2 = i
                break
        return (w_2 - w_1 + 1) / (h_2 - h_1 + 1)


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

    if split == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {split}')


def build(split, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided data path {root} does not exist'
    PATHS = {
        'train': (root / 'train' / 'backgrounds', root / 'train' / 'objects', root / 'train' / 'annotations.json'),
        'test': (root / 'test' / 'backgrounds', root / 'test' / 'objects', root / 'test' / 'annotations.json'),
    }
    img_folder, object_folder, ann_file = PATHS[split]
    dataset = Cityscapes(img_folder, object_folder, ann_file, split, transforms=make_cityscapes_transforms(split))
    return dataset

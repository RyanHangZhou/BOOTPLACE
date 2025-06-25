import os
import re
import imageio
import numpy as np
from PIL import Image, ImageFilter
from skimage.transform import resize, AffineTransform, warp

import torch
from torchvision.transforms import Compose

import datasets.transforms as T

# Constants
PATCH_SIZE = 64
CLASS_LIST = [
    'car', 'person', 'rider', 'train', 'bus', 'bicycle', 'truck', 'motorcycle',
    'car_r', 'person_r', 'rider_r', 'train_r', 'bus_r', 'bicycle_r', 'truck_r', 'motorcycle_r'
]

transform = T.Compose([
    # T.RandomResize([800], max_size=960),
    T.ToTensor(),
    T.Normalize([0.440, 0.440, 0.440, 0], [0.320, 0.319, 0.318, 0.5])
])

def ensure_dir(file_path):
    os.makedirs(file_path, exist_ok=True)

def sort_list_by_number_only(lst):
    def extract_numbers(s):
        return [float(x) for x in re.findall(r'\d+(?:\.\d+)?', s)]
    return sorted(lst, key=extract_numbers)

def extract_patches(patch_path):
    object_num = 100
    patch_list = sort_list_by_number_only(os.listdir(patch_path))[:object_num]
    patch_canvas = np.zeros((object_num * PATCH_SIZE, PATCH_SIZE, 4))
    patch_classes = np.zeros(object_num, int)

    for i, fname in enumerate(patch_list):
        patch_classes[i] = CLASS_LIST.index(fname[:-7])
        patch_img = resize(imageio.imread(os.path.join(patch_path, fname)), (PATCH_SIZE, PATCH_SIZE))
        patch_canvas[i * PATCH_SIZE:(i + 1) * PATCH_SIZE, :] = patch_img

    patch_image = Image.fromarray((patch_canvas * 255).astype(np.uint8))
    return patch_image, len(patch_list), patch_list

def extract_patches2(patch_path):
    patch_canvas = np.zeros((PATCH_SIZE, PATCH_SIZE, 4))
    patch_img = resize(imageio.imread(patch_path), (PATCH_SIZE, PATCH_SIZE))
    patch_canvas[:PATCH_SIZE, :] = patch_img
    patch_image = Image.fromarray((patch_canvas * 255).astype(np.uint8))
    return patch_image, 1, [patch_path]

def combine_list(patches_gt, patches, patch_num_gt, patch_num):
    patches_gt = patches_gt.crop((0, 0, patches_gt.width, patch_num_gt * patches_gt.width))
    patches = patches.crop((0, 0, patches.width, patch_num * patches.width))
    dst = Image.new('RGBA', (patches_gt.width, patches_gt.height + patches.height))
    dst.paste(patches_gt, (0, 0))
    dst.paste(patches, (0, patches_gt.height))
    return dst, patch_num_gt + patch_num

def get_aspect_ratio(patch):
    h, w = patch.shape[:2]
    nonzero = np.argwhere(patch[:, :, 0] > 0)
    y1, x1 = nonzero.min(axis=0)
    y2, x2 = nonzero.max(axis=0)
    return (x2 - x1 + 1) / (y2 - y1 + 1)

def image_composition(image, image_vis, patch, bbx, is_top_left=False, is_square=True, is_mask=False):
    img_h, img_w = image.shape[:2]

    mask = patch[:, :, 3][:, :, None]
    obj = patch[:, :, :3] * (mask / 255.0)
    p_s = patch.shape[0]

    if not is_top_left:
        bbx = bbx.cpu().detach().numpy()
        center_w = int(bbx[0] * img_w)
        center_h = int(bbx[1] * img_h)
        scale = int(((bbx[3] * img_h + bbx[2] * img_w) / 2.0) / 2) * 2
        asp_ratio = get_aspect_ratio(patch)
        scale_w = int(bbx[2] * img_w / 2) * 2
        scale_h = int(bbx[3] * img_h / 2) * 2
        scale_ = scale_w if asp_ratio > (scale_w / scale_h) else scale_h
    else:
        new_p_s = 128
        center_w = int(new_p_s / 2)
        center_h = int(new_p_s / 2)
        scale = new_p_s

    tmp_patch_canvas = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    tmp_mask_canvas = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
    tmp_patch_canvas[:patch.shape[0], :patch.shape[1], :] = obj
    tmp_mask_canvas[:patch.shape[0], :patch.shape[1], :] = mask

    if is_square:
        tform = AffineTransform(scale=(scale / p_s, scale / p_s),
                                translation=(center_w - scale / 2, center_h - scale / 2))
    else:
        tform = AffineTransform(scale=scale_ / p_s,
                                translation=(center_w - scale_ / 2, center_h - scale_ / 2))

    tmp_patch_canvas = warp(tmp_patch_canvas, tform.inverse)
    tmp_mask_canvas = warp(tmp_mask_canvas, tform.inverse) / 255.0

    image_out = image * (1 - tmp_mask_canvas) + tmp_patch_canvas * tmp_mask_canvas
    image_out_vis = image_vis * (1 - tmp_mask_canvas) + tmp_patch_canvas * tmp_mask_canvas + 100 * tmp_mask_canvas * [0, 0, 1]

    if is_mask:
        return np.clip(image_out, 0, 255).astype(np.uint8), \
               np.clip(image_out_vis, 0, 255).astype(np.uint8), \
               np.clip(tmp_mask_canvas * 255.0, 0, 255).astype(np.uint8)
    else:
        return np.clip(image_out, 0, 255).astype(np.uint8), \
               np.clip(image_out_vis, 0, 255).astype(np.uint8)

def intersection(boxes, box):
    intersection_mat = torch.zeros(len(boxes), dtype=torch.bool)
    for i, b in enumerate(boxes):
        x1, y1, w1, h1 = b
        x2, y2, w2, h2 = box
        xa1, ya1 = max(x1 - w1 / 2, x2 - w2 / 2), max(y1 - h1 / 2, y2 - h2 / 2)
        xa2, ya2 = min(x1 + w1 / 2, x2 + w2 / 2), min(y1 + h1 / 2, y2 + h2 / 2)
        inter_area = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        if inter_area / union_area > 0.2:
            intersection_mat[i] = True
    return intersection_mat

def load_one(data_dir, filename):
    img_h, img_w = 540, 960
    im = Image.open(os.path.join(data_dir['im'], filename)).convert("RGB").resize((img_w, img_h))
    imm = Image.open(os.path.join(data_dir['im'], filename)).convert("RGB")
    image_np = np.array(im)
    imagem_np = np.array(imm)
    print('filename: ', filename)
    print('1st line: ', image_np[0, 0:10, 0])
    print('1st line: ', imagem_np[0, 0:10, 0])
    im_inpaint = Image.open(os.path.join(data_dir['im_inpaint'], filename)).convert("RGB").resize((img_w, img_h))
    patches, patch_num, patch_list = extract_patches2(os.path.join(data_dir['patch'], filename))
    im_t, _, _ = transform(im, patches, None)
    print('1st line im_t: ', im_t[0, 0:10, 0])
    patches_np = np.array(patches)
    print('16x16 patch: ', patches_np[16:32, 16:32, 0])
    im_inpaint_t, patches_t, _ = transform(im_inpaint, patches, None)
    print(np.shape(patches_t))
    print('1st line patch im_t: ', patches_t[0, 0:10, 0])
    print('16x16 patch_t: ', patches_t[0, 16:32, 16:32])
    return {
        'im': im,
        'im_t': im_t,
        'im_inpaint': im_inpaint,
        'im_inpaint_t': im_inpaint_t,
        'patches': patches,
        'patches_t': patches_t,
        'patch_num': patch_num,
        'patch_list': patch_list
    }

def load_all(data_dir, filename):
    img_h, img_w = 540, 960
    im = Image.open(os.path.join(data_dir['im'], filename[:-4] + '.jpg')).convert("RGB").resize((img_w, img_h))
    patches, patch_num, patch_list = extract_patches2(os.path.join(data_dir['patch'], filename))
    im_t, patches_t, _ = transform(im, patches, None)
    return {
        'im': im,
        'im_t': im_t,
        'patches': patches,
        'patches_t': patches_t,
        'patch_num': patch_num,
        'patch_list': patch_list
    }

def load_all2(data_dir, filename):
    img_h, img_w = 540, 960
    im = Image.open(os.path.join(data_dir['im'], filename + '.png')).convert("RGB").resize((img_w, img_h))
    patches, patch_num, patch_list = extract_patches(os.path.join(data_dir['patch'], filename))
    im_t, patches_t, _ = transform(im, patches, None)
    return {
        'im': im,
        'im_t': im_t,
        'patches': patches,
        'patches_t': patches_t,
        'patch_num': patch_num,
        'patch_list': patch_list
    }

import datasets.transforms as T
import imageio
import numpy as np
import os
import re

from PIL import Image, ImageFilter
from skimage.transform import resize, AffineTransform, warp


patch_s = 64
class_list = ['car', 'person', 'rider', 'train', 'bus', 'bicycle', 'truck', 'motorcycle', 
              'car_r', 'person_r', 'rider_r', 'train_r', 'bus_r', 'bicycle_r', 'truck_r', 'motorcycle_r']

transform = T.Compose([
    T.RandomResize([800], max_size=960),
    T.ToTensor(),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    T.Normalize([0.440, 0.440, 0.440, 0], [0.320, 0.319, 0.318, 0.5])
])


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def sort_list_by_number_only(lst):
    def extract_numbers(sample):
        return [float(s) for s in re.findall(r'\d+(?:\.\d+)?', sample)]
    return sorted(lst, key=extract_numbers)

    
def extract_patches(patch_path):
    object_num = 100

    patch_list = os.listdir(patch_path)
    patch_list = sort_list_by_number_only(patch_list)[:object_num]
    counter = 0
    patch_canvas = np.zeros((object_num*patch_s, patch_s, 4))
    patch_classes = np.zeros((object_num), np.int)
    for location_file in patch_list:
        patch_classes[counter] = class_list.index(location_file[:-7])
        patch_sample_path = os.path.join(patch_path, location_file)
        patch_sample = imageio.imread(patch_sample_path)
        patch_sample = resize(patch_sample, (patch_s, patch_s))
        patch_canvas[counter*patch_s:(counter+1)*patch_s, :] = patch_sample
        counter += 1
    patch_canvas = Image.fromarray(np.uint8(patch_canvas*255))
    return patch_canvas, counter, patch_list

def extract_patches2(patch_path):
    object_num = 100

    patch_list = patch_path
    # patch_list = sort_list_by_number_only(patch_list)[:object_num]
    counter = 0
    patch_canvas = np.zeros((object_num*patch_s, patch_s, 4))
    patch_sample = imageio.imread(patch_path)
    patch_sample = resize(patch_sample, (patch_s, patch_s))
    patch_canvas[counter*patch_s:(counter+1)*patch_s, :] = patch_sample
    counter += 1
    patch_canvas = Image.fromarray(np.uint8(patch_canvas*255))
    return patch_canvas, counter, [patch_list]



def combine_list(patches_gt, patches, patch_num_gt, patch_num):
    img_s, img_h = patches_gt.size[:2]
    patches_gt = patches_gt.crop((0, 0, img_s, patch_num_gt*img_s))
    patches = patches.crop((0, 0, img_s, patch_num*img_s))
    dst = Image.new('RGBA', (patches_gt.width, patches_gt.height + patches.height))
    dst.paste(patches_gt, (0, 0))
    dst.paste(patches, (0, patches_gt.height))
    return dst, patch_num_gt + patch_num


def get_aspect_ratio(patch):
    """ Get aspect ratio of a patch """

    img_h, img_w = patch.shape[:2]

    for i in range(img_h):
        line = patch[i, :, 0]
        if(np.sum(line) > 0):
            h_1 = i
            break
    for i in range(img_h-1, 0, -1):
        line = patch[i, :, 0]
        if(np.sum(line) > 0):
            h_2 = i
            break
    for i in range(img_w):
        column = patch[:, i, 0]
        if(np.sum(column) > 0):
            w_1 = i
            break
    for i in range(img_w-1, 0, -1):
        column = patch[:, i, 0]
        if(np.sum(column) > 0):
            w_2 = i
            break
    return (w_2 - w_1 + 1) / (h_2 - h_1 + 1)


def image_composition(image, image_vis, patch, bbx, is_top_left=False, is_square=True, is_mask=False): 
    img_h, img_w = image.shape[:2]

    mask = patch[:, :, 3][:, :, None]
    obj = patch[:, :, :3]*(mask/255)
    p_s = patch.shape[0]

    if not is_top_left: 
        bbx = bbx.cpu().detach().numpy()
        center_w = int(bbx[0]*img_w)
        center_h = int(bbx[1]*img_h)
        scale = int(((bbx[3]*img_h + bbx[2]*img_w)/2.0)/2)*2
        # scale = int(np.sqrt((bbx[3]*img_h) * (bbx[2]*img_w))/2)*2
        asp_ratio = get_aspect_ratio(patch)
        # print(asp_ratio)
        # ssss
        scale_w = int(bbx[2]*img_w/2)*2
        scale_h = int(bbx[3]*img_h/2)*2
        if asp_ratio>scale_w / scale_h:
            scale_ = scale_w
        else:
            scale_ = scale_h
    elif is_top_left:
        new_p_s = 128 # scale of object patch for visualization
        center_w = int(new_p_s / 2)
        center_h = int(new_p_s / 2)
        scale = new_p_s
        

    tmp_patch_canvas = np.zeros((image.shape[0], image.shape[1], 3))
    tmp_mask_canvas = np.zeros((image.shape[0], image.shape[1], 1))
    tmp_patch_canvas[:patch.shape[0], :patch.shape[1], :] = obj
    tmp_mask_canvas[:patch.shape[0], :patch.shape[1], :] = mask

    if is_square:
        tform = AffineTransform(scale=(scale/p_s, scale/p_s), translation=(center_w-scale/2, center_h-scale/2))
    else:
        # tform = AffineTransform(scale=(scale_w/p_w, scale_h/p_h), translation=(center_w-scale_w/2, center_h-scale_h/2))
        tform = AffineTransform(scale=scale_/p_s, translation=(center_w-scale_/2, center_h-scale_/2))
    tmp_patch_canvas = warp(tmp_patch_canvas, tform.inverse)
    tmp_mask_canvas = warp(tmp_mask_canvas, tform.inverse)
    tmp_mask_canvas = tmp_mask_canvas/255.

    image_out = image*(1-tmp_mask_canvas) + tmp_patch_canvas*tmp_mask_canvas
    image_out_vis = image_vis*(1-tmp_mask_canvas) + tmp_patch_canvas*tmp_mask_canvas + 100*tmp_mask_canvas*[0, 0, 1]
    
    if is_mask: 
        return np.clip(image_out, 0, 255).astype(int), np.clip(image_out_vis, 0, 255).astype(int), np.clip(tmp_mask_canvas*255., 0, 255).astype(int)
    else:
        return np.clip(image_out, 0, 255).astype(int), np.clip(image_out_vis, 0, 255).astype(int)


def intersection(boxes, box):
    intersection_mat = torch.zeros((boxes.shape[0]))
    for i in range(boxes.shape[0]):
        box_i = boxes[i]
        x1, y1, w1, h1 = box_i
        x2, y2, w2, h2 = box

        x_intersection = torch.maximum(x1-w1/2, x2-w2/2)
        y_intersection = torch.maximum(y1-h1/2, y2-h2/2)
        x_end_intersection = torch.minimum(x1 + w1/2, x2 + w2/2)
        y_end_intersection = torch.minimum(y1 + h1/2, y2 + h2/2)

        # Calculate the area of the intersection rectangle
        intersection_area = max(0, x_end_intersection - x_intersection) * max(0, y_end_intersection - y_intersection)
        area_box1 = w1 * h1
        area_box2 = w2 * h2

        # Calculate the Union area by subtracting the intersection area
        # from the sum of the areas of the two bounding boxes
        union_area = area_box1 + area_box2 - intersection_area

        # Calculate the IoU
        iou = intersection_area / union_area
        if iou > 0.2:
            intersection_mat[i] = 1
    return intersection_mat.bool()


def load_one(data_dir, filename):
    """ Load scene image, background image, patches to compose """

    img_h, img_w = 540, 960
    im = Image.open(os.path.join(data_dir['im'], filename)).convert("RGB").resize((img_w, img_h))
    im_inpaint = Image.open(os.path.join(data_dir['im_inpaint'], filename)).convert("RGB").resize((img_w, img_h))
    patches, patch_num, patch_list = extract_patches2(os.path.join(data_dir['patch'], filename))
    # patches_gt, patch_num_gt, patch_list_gt = extract_patches(os.path.join(data_dir['gt_patch'], filename)[:-4])

    # patches, patch_num = combine_list(patches_gt, patches, patch_num_gt, patch_num)
    # patch_list = patch_list_gt + patch_list

    # median_radius = 13 # 9
    # im_t = im.filter(ImageFilter.MedianFilter(median_radius))
    # im_inpaint_t = im_inpaint.filter(ImageFilter.MedianFilter(median_radius))

    # radius = 5
    # im_t = im.filter(ImageFilter.GaussianBlur(radius))
    # im_inpaint_t = im_inpaint.filter(ImageFilter.GaussianBlur(radius))
    im_t = im
    im_inpaint_t = im_inpaint

    im_t, _, _ = transform(im_t, patches, None)
    im_inpaint_t, patches_t, _ = transform(im_inpaint_t, patches, None)

    data_output = {}
    data_output['im'] = im
    data_output['im_t'] = im_t
    data_output['im_inpaint'] = im_inpaint
    data_output['im_inpaint_t'] = im_inpaint_t
    data_output['patches'] = patches
    data_output['patches_t'] = patches_t
    # data_output['patch_num_gt'] = patch_num_gt
    data_output['patch_num'] = patch_num
    data_output['patch_list'] = patch_list

    return data_output

def load_all(data_dir, filename):
    """ Load scene image, and patches to compose """

    img_h, img_w = 540, 960
    im = Image.open(os.path.join(data_dir['im'], filename[:-4]+'.jpg')).convert("RGB").resize((img_w, img_h))
    patches, patch_num, patch_list = extract_patches2(os.path.join(data_dir['patch'], filename))

    # median_radius = 13 # 9
    # im_t = im.filter(ImageFilter.MedianFilter(median_radius))

    # radius = 5
    # im_t = im.filter(ImageFilter.GaussianBlur(radius))
    
    im_t = im

    im_t, patches_t, _ = transform(im_t, patches, None)

    data_output = {}
    data_output['im'] = im
    data_output['im_t'] = im_t
    data_output['patches'] = patches
    data_output['patches_t'] = patches_t
    data_output['patch_num'] = patch_num
    data_output['patch_list'] = patch_list
    
    return data_output

def load_all2(data_dir, filename):
    """ Load scene image, and patches to compose """

    img_h, img_w = 540, 960
    im = Image.open(os.path.join(data_dir['im'], filename + '.png')).convert("RGB").resize((img_w, img_h))
    patches, patch_num, patch_list = extract_patches(os.path.join(data_dir['patch'], filename))
    im_t, patches_t, _ = transform(im, patches, None)

    data_output = {}
    data_output['im'] = im
    data_output['im_t'] = im_t
    data_output['patches'] = patches
    data_output['patches_t'] = patches_t
    data_output['patch_num'] = patch_num
    data_output['patch_list'] = patch_list
    
    return data_output


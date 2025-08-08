from glob import glob
import scipy.io as sio
from pycocotools import mask as cocomask
from PIL import Image
import os
import glob
import numpy as np
import re
import imageio
from skimage.transform import resize
import json

MAX_N = 10
object_num = 50
patch_s = 64

class_list = ['car', 'person', 'rider', 'train', 'bus', 'bicycle', 'truck', 'motorcycle']


def sort_list_by_number_only(lst):
    def extract_numbers(sample):
        return [float(s) for s in re.findall(r'\d+(?:\.\d+)?', sample)]
    return sorted(lst, key=extract_numbers)

def extract_location_mat(location_path, patch_list):
    
    def extract_floats_from_file(filename):

        floats = []
        with open(filename, 'r') as file:
            for line in file:
                try:
                    number = int(line)
                    floats.append(number)
                except ValueError:
                    pass
        return floats

    location_mat = np.ones((object_num, 4), dtype=float) # initialize empty token with 0.5
    location_mat = np.tile(location_mat, (object_num, 1))
    location_list = os.listdir(location_path)
    location_list = sort_list_by_number_only(location_list)[:object_num]
    counter = 0

    for location_file in patch_list:
        location_sample_path = os.path.join(location_path, location_file[:-4]+'.txt')
        location_sample = extract_floats_from_file(location_sample_path)
        location_mat[counter, :] = location_sample
        counter += 1
    
    return location_mat[:len(location_list), :], len(location_list)

def extract_patches(patch_path):
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
    
    patch_canvas = np.transpose(patch_canvas, [2, 0, 1])

    return patch_canvas.astype(np.float32), patch_classes, len(patch_list), patch_list

categories = [
    {
        "supercategory": "car",
        "name": "car",
        "id": 0
    },
    {
        "supercategory": "person",
        "name": "person",
        "id": 1
    },
    {
        "supercategory": "rider",
        "name": "rider",
        "id": 2
    },
    {
        "supercategory": "train",
        "name": "train",
        "id": 3
    },
    {
        "supercategory": "bus",
        "name": "bus",
        "id": 4
    },
    {
        "supercategory": "bicycle",
        "name": "bicycle",
        "id": 5
    },
    {
        "supercategory": "truck",
        "name": "truck",
        "id": 6
    },
    {
        "supercategory": "motorcycle",
        "name": "motorcycle",
        "id": 7
    },
]

NAME_TO_CATEGORY_ID = {
    "car": 0,
    "person": 1,
    "rider": 2,
    "train": 3,
    "bus": 4,
    "bicycle": 5,
    "truck": 6,
    "motorcycle": 7
}

# root = 'data/Cityscapes_large'
# root = 'data/Cityscapes_clean'
root = 'data/Cityscapes'
phases = ['train', 'test']
# phases = ['test']
# phases = ['train']
for phase in phases:
    ''' 1. Inputs and output dir'''
    original_image_path = os.path.join(root, phase, 'original_image')
    patch_path = os.path.join(root, phase, 'patch')
    location_path = os.path.join(root, phase, 'location')
    json_path = os.path.join(root, phase, 'annotations.json')
    inpainted_image_path = os.path.join(root, phase, 'inpainted_image') #####
    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    ''' 2. Usable sample list '''
    patch_list = glob.glob(patch_path + '/*')
    sample_dir_list = []
    for i in patch_list:
        if(len(os.listdir(i))>0):
            sample_dir_list.append(i)
    sample_list = [x.split('/')[-1] for x in sample_dir_list]
    inpainted_list = os.listdir(inpainted_image_path) ####
    inpainted_list = [s[:-4] for s in inpainted_list] ####

    annot_count = 0
    image_id = 0
    processed = 0
    for j in range(len(sample_list)):
        sample = sample_list[j]
        if sample in inpainted_list: ####

            ''' 3. Load image info '''
            img = Image.open(os.path.join(original_image_path, sample+'.png'))
            filename = sample+'.png'
            img_w, img_h = img.size
            img_elem = {
                "file_name": filename, 
                "height": img_h,
                "width": img_w,
                "id": image_id
            }
            res_file["images"].append(img_elem)

            ''' 4. Load patches and locations (unormalized) '''
            patch_canvas, patch_classes, patch_num, patch_list = extract_patches(os.path.join(patch_path, sample))
            location_mat = extract_location_mat(os.path.join(location_path, sample), patch_list)[0]

            """ 5. Get (x0, y0, w, h), where (x0, y0) is the left top coordinate """
            for i in range(patch_num):
                location_tmp = location_mat[i]
                xmin = int(location_tmp[0])
                ymin = int(location_tmp[1])
                xmax = int(location_tmp[2])
                ymax = int(location_tmp[3])
                w = xmax - xmin
                h = ymax - ymin
                area = w * h
                poly = [[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax]
                ]

                annot_elem = {
                    "id": annot_count,
                    "bbox": [
                        float(xmin),
                        float(ymin),
                        float(w),
                        float(h)
                    ],
                    "segmentation": list([poly]),
                    "image_id": image_id,
                    "category_id": NAME_TO_CATEGORY_ID[patch_list[i][:-7]],
                    "area": float(area),
                    "object_name": patch_list[i][:-4],
                    "iscrowd": 0
                }

                res_file["annotations"].append(annot_elem)
                annot_count += 1
            
            image_id += 1
            processed += 1
        
        """ 6. Write into json file """
        with open(json_path, "w") as f:
            json_str = json.dumps(res_file)
            f.write(json_str)
        
        print("Processed {} {} images...".format(processed, phase))

    print("Done.")

import os, sys
import torch, json
import numpy as np
import argparse

from models import build_model
import main
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops
import re
import imageio
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
import datasets.transforms as T
from util.test_utils import ensure_dir, get_aspect_ratio, load_one, load_all, image_composition
from pycocotools.coco import COCO
from thop import profile
import time

class_list = ['car', 'person', 'rider', 'train', 'bus', 'bicycle', 'truck', 'motorcycle']
class_map = {'car': 0, 'person': 1, 'rider': 2, 'train': 3, 'bus': 4, 'bicycle': 5, 'truck': 6, 'motorcycle': 7}
patch_s = 64

transform = T.Compose([
    T.RandomResize([800], max_size=960),
    T.ToTensor(),
    T.Normalize([0.440, 0.440, 0.440, 0], [0.320, 0.319, 0.318, 0.5])
])

def box_xyxy_to_x1y1wh(xyxy_boxes):
    x1 = xyxy_boxes[:, 0]
    y1 = xyxy_boxes[:, 1]
    x2 = xyxy_boxes[:, 2]
    y2 = xyxy_boxes[:, 3]
    width = x2 - x1
    height = y2 - y1
    x1y1wh_boxes = torch.stack((x1, y1, width, height), dim=1)

    return x1y1wh_boxes

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_clip', default=1, type=float,
                        help="clip coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--ce_loss_coef', default=1, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--clip_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco') # 
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--is_mask', action='store_true')
    parser.add_argument('--pred_threshold', default=0.7, type=float)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--pretrained_model', type=str, help='path of pretrained model')
    parser.add_argument('--im_root', type=str, help='path of test set')
    parser.add_argument('--is_recompose', default=False, help='recompose or compose')
    parser.add_argument('--is_one', default=False, help='compose one or multiple')
    parser.add_argument('--savedir', default='', help='save directory')
    parser.add_argument('--num_query', type=int, default=200, help='threshold of query number')
    
    
    return parser


''' 1. Load model '''
parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
args.device = 'cuda' 
model, criterion, postprocessors = build_model(args)
checkpoint = torch.load(args.pretrained_model, map_location='cpu')
model.load_state_dict(checkpoint['model'])


''' 2. Load Cityscapes category names: (8*2+1) classes '''
with open('util/Cityscapes_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}


''' 3. Parameter settings '''
is_unique = True
num_query = args.num_query if is_unique else 10
is_recompose = args.is_recompose
is_one = args.is_one
pred_threshold = args.pred_threshold
if not is_recompose:
    im_root = os.path.join(args.im_root, 'backgrounds_single')
    # im_root = os.path.join(args.im_root, 'inpaint')
    im_inpaint_root = os.path.join(args.im_root, 'backgrounds_single')
    patch_root = os.path.join(args.im_root, 'objects_single')
    gt_patch_root = os.path.join(args.im_root, 'patch_gt')
    location_root = os.path.join(args.im_root, 'location_single')
    location_scene_root = os.path.join(args.im_root, 'location')
else:
    # im_root = os.path.join(args.im_root, 'inpaint')
    # patch_root = os.path.join(args.im_root, 'patch_gt_clean')
    # patch_root = os.path.join(args.im_root, 'patch_gt___')
    im_root = os.path.join(args.im_root, 'inpainted_image')
    patch_root = os.path.join(args.im_root, 'objects_single')

savedir = args.savedir
ensure_dir(savedir)


''' 4. Placement prediction '''
counter = 0
patch_list = os.listdir(patch_root)
for filename in patch_list[0:]:
    print(filename)
    # import pdb; pdb.set_trace()
# try: 

    ''' 4.1 Read images and patches '''
    if not is_recompose:
        data_dir = {}
        data_dir['im'] = im_root
        data_dir['patch'] = patch_root
        data_dir['im_inpaint'] = im_inpaint_root
        data_dir['gt_patch'] = gt_patch_root

        data_output = load_one(data_dir, filename)

        im = data_output['im']
        im_t = data_output['im_t']
        im_inpaint = data_output['im_inpaint']
        im_inpaint_t = data_output['im_inpaint_t']
        patches = data_output['patches']
        patches_t = data_output['patches_t']
        # patch_num_gt = data_output['patch_num_gt']
        patch_num = data_output['patch_num']
        patch_list = data_output['patch_list']
    else:
        data_dir = {}
        data_dir['im'] = im_root
        data_dir['patch'] = patch_root

        data_output = load_all(data_dir, filename)

        im = data_output['im']
        im_t = data_output['im_t']
        patches = data_output['patches']
        patches_t = data_output['patches_t']
        patch_num = data_output['patch_num']
        patch_list = data_output['patch_list']

        im_inpaint_t = im_t
    
    # ''' GT placement '''
    json_path = 'data/Cityscapes/test/annotations.json'
    coco = COCO(json_path)

    num_images = len(coco.getImgIds())
    print(f"Number of images: {num_images}") # 166
    h, w = im_t.shape[1:3]

    for ids in range(166):
        image_info = coco.loadImgs(ids)[0]
        image_id = image_info["file_name"][:-4]
        # print(image_id, filename)

        if image_id == filename[:-4]:
            anno = coco.loadAnns(coco.getAnnIds(imgIds=[ids]))
            boxes_gt = [obj["bbox"] for obj in anno]
            object_name = [obj["object_name"] for obj in anno]
            boxes_gt = torch.as_tensor(boxes_gt, dtype=torch.float32).reshape(-1, 4)
            boxes_gt[:, 2:] += boxes_gt[:, :2]
            boxes_gt[:, 0::2].clamp_(min=0, max=w)
            boxes_gt[:, 1::2].clamp_(min=0, max=h)
            boxes_gt = box_xyxy_to_x1y1wh(boxes_gt)
            boxes_gt = boxes_gt / torch.tensor([w, h, w, h], dtype=torch.float32)
            object_name = [obj["object_name"] for obj in anno]
            break
        
    # print(np.shape(boxes_gt))
    # print(object_name)
    # print(patch_root)
    # print(os.listdir(os.path.join(patch_root, filename[:-4])))
    # print(boxes_gt)
    # print(np.shape(boxes_gt))

    boxes_gt = []
    location_scene_path = os.path.join(location_scene_root, filename[:-4])
    for l in os.listdir(location_scene_path):
        with open(os.path.join(location_scene_path, l), "r") as file:
            values = file.readlines()
        w1, h1, w2, h2 = map(float, values)
        # boxes_gt.append([w1, h1, w2, h2]/ [w, h, w, h])
        boxes_gt.append(np.array([w1, h1, w2, h2]) / np.array([w, h, w, h]))

    # print(boxes_gt)
    # print(np.shape(boxes_gt))
    



    with open(os.path.join(location_root, filename[:-4]+'.txt'), "r") as file:
        values = file.readlines()
    w1, h1, w2, h2 = map(float, values)
    w1 = w1/960
    w2 = w2/960
    h1 = h1/540
    h2 = h2/540
    boxes_target = [w1, h1, w2, h2]
    # print(f"h1: {h1}, w1: {w1}, h2: {h2}, w2: {w2}")

    boxes_gt = np.array([row for row in boxes_gt if not np.array_equal(row, boxes_target)])
    boxes_gt = torch.tensor(boxes_gt, dtype=torch.float32)
    # print(boxes_gt)
    # print(np.shape(boxes_gt))

    # import pdb; pdb.set_trace()



    # import pdb; pdb.set_trace()
    ''' 4.2 Model prediction '''
    # 4.2.1 output stuffs
    # print(np.shape(boxes_gt))
    # print(np.shape(patches_t))
    # boxes_gt = boxes_gt[None, :]
    targets = [{}]
    targets[0]['boxes'] = boxes_gt
    targets[0]['negative_bin_mask'] = torch.zeros(0)
    # targets[0]['boxes'] = torch.zeros((20, 4))
    # sss
    start = time.clock()
    model_output = model.cuda()(im_t[None].cuda(), patches_t[None].cuda(), targets)
    end = time.clock()
    print(end-start)


    
    # flops, params = profile(model, inputs=(im_t[None].cuda(), patches_t[None].cuda(), targets))
    
    # print(flops)
    # print(params)
    
    # www

    probas = model_output['pred_logits'][0, :, :].softmax(-1)[:, :-1] # [200, 8]
    probas_obj = 1 - model_output['pred_logits'][0, :, :].softmax(-1)[:, -1] # [200, 8]
    # probas = model_output['pred_logits'][0, :, :][:, :-1]
    boxes = model_output['pred_boxes'][0, :, :] # [200, 4] (cxcywh), [-1, 1]
    clip1 = model_output['pred_clip'][0, :, :patch_num] # [200, 50] -> [200, patch_num]
    clip2 = model_output['pred_clip2'][0, :patch_num, :].T # [50, 200] -> [200, patch_num]
    clip = (clip1.softmax(dim=0) + clip2.softmax(dim=0))/2 # [200, patch_num]
    # clip = (clip1 + clip2)/2 # [200, patch_num]

    probas_car = probas[:, 0]
    probas_pedestrian = probas[:, 1]
    # probas_car_real = probas[:, 8]
    # probas_pedestrian_real = probas[:, 9]
    bbx_area = boxes[:, 2] * boxes[:, 3]
    keep_area = bbx_area > 0.08 #0.15
    
    scores, labels = probas.max(-1)
    # scores_fake, labels_fake = probas[:, :8].max(-1)
    # scores_real, labels_real = probas[:, 8:].max(-1)

    img_h, img_w = im_t.shape[1], im_t.shape[2]
    vslzr = COCOVisualizer()


    # 4.2.2 
    # keep_max_class = probas.max(-1).values > pred_threshold # [200]
    keep_all_category = probas > pred_threshold # [200, 8]
    # keep_real = probas_car_real > 0.3
    # keep_car = probas[:, 0] > pred_threshold # [200]
    # keep_pedestrian = probas[:, 1] > pred_threshold # [200]
    # print(torch.sum(keep))

    threshold = 0.7
    # keep_obj = scores_real > threshold
    # keep_obj = keep_obj.bool()
    # keep_obj = torch.sum(keep_obj, -1)


    if not is_recompose and is_unique:

        patch_target = imageio.imread(os.path.join(patch_root, filename))

        ''' 4.3.0 Topk prediction '''
        threshold = 0.1
        keep = scores > threshold
        scores_topk = scores[keep]
        box_label = [id2name[int(item)] for item in labels[keep]]

        pred_dict = {'boxes': boxes[keep], 'size': torch.Tensor([img_h, img_w]), 'box_label': box_label, 'image_id': 0}
        visual_dir = os.path.join(savedir, '0_visualize_topk')
        ensure_dir(visual_dir)
        vslzr.savefig(im_t, pred_dict, filename=filename[:-4], savedir=visual_dir, dpi=500)


        # ''' 4.3.1 Topk real car prediction '''
        # threshold = 0.7
        # keep_car = probas_car_real > threshold
        # scores_car = probas_car_real[keep_car]
        # box_label = [str(round(item, 3)) for item in scores_car.cpu().detach().numpy()]

        # pred_dict = {'boxes': boxes[keep_car], 'size': torch.Tensor([img_h, img_w]), 'box_label': box_label, 'image_id': 0}
        # visual_dir = os.path.join(savedir, '1_visualize_topk_real_car')
        # ensure_dir(visual_dir)
        # vslzr.savefig(im_t, pred_dict, filename=filename[:-4], savedir=visual_dir, dpi=500)


        ''' 4.3.2 Topk proposal car prediction '''
        num_query = 10 #50
        logits_topk, idx_topk = torch.topk(probas_car, num_query, dim=0)
        keep_car_ = torch.zeros_like(probas_car).bool()
        keep_car_[idx_topk] = 1
        # threshold = 0.1
        # keep_car_ = probas_car* ~keep_car > threshold
        # keep_car_ = probas_car > threshold
        scores_car = probas_car[keep_car_]
        box_label = [str(round(item, 3)) for item in scores_car.cpu().detach().numpy()]

        pred_dict = {'boxes': boxes[keep_car_], 'size': torch.Tensor([img_h, img_w]), 'box_label': box_label, 'image_id': 0}
        visual_dir = os.path.join(savedir, '2_visualize_topk_proposal_car')
        ensure_dir(visual_dir)
        vslzr = COCOVisualizer()
        vslzr.savefig(im_t, pred_dict, filename=filename[:-4], savedir=visual_dir, dpi=500)


        # ''' 4.3.3 Topk real pedestrian prediction '''
        # # num_query = 20
        # # logits_topk, idx_topk = torch.topk(probas_pedestrian_real, num_query, dim=0)
        # # keep_pedestrian = torch.zeros_like(probas_pedestrian_real).bool()
        # # keep_pedestrian[idx_topk] = 1
        # threshold = 0.7
        # keep_pedestrian = probas_pedestrian_real > threshold
        # scores_pedestrian = probas_pedestrian_real[keep_pedestrian]
        # box_label = [str(round(item, 3)) for item in scores_pedestrian.cpu().detach().numpy()]

        # pred_dict = {'boxes': boxes[keep_pedestrian], 'size': torch.Tensor([img_h, img_w]), 'box_label': box_label, 'image_id': 0}
        # visual_dir = os.path.join(savedir, '3_visualize_topk_real_pedestrian')
        # ensure_dir(visual_dir)
        # vslzr.savefig(im_t, pred_dict, filename=filename[:-4], savedir=visual_dir, dpi=500)


        ''' 4.3.4 Topk proposal pedestrian prediction '''
        num_query = 10 #50
        logits_topk, idx_topk = torch.topk(probas_pedestrian, num_query, dim=0)
        keep_pedestrian_ = torch.zeros_like(probas_pedestrian).bool()
        keep_pedestrian_[idx_topk] = 1
        scores_pedestrian = probas_pedestrian[keep_pedestrian_]
        box_label = [str(round(item, 3)) for item in scores_pedestrian.cpu().detach().numpy()]

        pred_dict = {'boxes': boxes[keep_pedestrian_], 'size': torch.Tensor([img_h, img_w]), 'box_label': box_label, 'image_id': 0}
        visual_dir = os.path.join(savedir, '4_visualize_topk_proposal_pedestrian')
        ensure_dir(visual_dir)
        vslzr.savefig(im_t, pred_dict, filename=filename[:-4], savedir=visual_dir, dpi=500)


        ''' 4.3.5 Topk proposal car prediction + aspect ratio '''
        asp_ratio = get_aspect_ratio(patch_target)
        box_asp_ratio = boxes[:, 2] / boxes[:, 3] * img_w / img_h

        num_query = 10 # 40
        logits_topk, idx_topk = torch.topk(-torch.abs(box_asp_ratio * keep_car_ - asp_ratio), num_query, dim=0)
        keep_asp_ratio = torch.zeros_like(probas_car).bool()
        keep_asp_ratio[idx_topk] = 1
        
        composed_im = np.asarray(im).astype(int)
        composed_im_vis = composed_im
        try:        
            composed_im, composed_im_vis = image_composition(composed_im, composed_im_vis, patch_target, boxes[keep_car_[0]], is_top_left=True)
        except:
            print('ssss') 
        
        scores_car_asp = probas_car[keep_asp_ratio]
        box_label = [str(round(item, 3)) for item in scores_car_asp.cpu().detach().numpy()]

        pred_dict = {'boxes': boxes[keep_asp_ratio], 'size': torch.Tensor([img_h, img_w]), 'box_label': box_label, 'image_id': 0}
        visual_dir = os.path.join(savedir, '5_visualize_topk_proposal_asp_car')
        ensure_dir(visual_dir)
        composed_im_vis = Image.fromarray(np.uint8(composed_im_vis))
        composed_im_vis_t, _, _ = transform(composed_im_vis, patches, None)
        vslzr.savefig(composed_im_vis_t, pred_dict, filename=filename[:-4], savedir=visual_dir, dpi=500)

        
        ''' 4.3.6 Topk proposal car prediction + aspect ratio + CLIP '''
        num_query = 30
        keep_bbx_large = bbx_area > 0.01
        print('keep_bbx_large: ', torch.sum(keep_bbx_large))
        # clip_thr = clip[:, 0] * probas_car * keep_asp_ratio * keep_bbx_large
        clip_thr =  probas_car * keep_asp_ratio * keep_bbx_large / clip[:, 0]
        logits_topk, idx_topk = torch.topk(clip_thr, num_query, dim=0)
        clip_picked = clip[:, 0][idx_topk]

        composed_im = np.asarray(im).astype(int)
        composed_im_vis = composed_im
        if torch.sum(keep_bbx_large) > 0: 
            try:        
                composed_im, composed_im_vis, composed_mask = image_composition(composed_im, composed_im_vis, patch_target, boxes[idx_topk[0]], is_square=False, is_mask=True)
            except:
                print('ssss')
        # try:        
        #     composed_im, composed_im_vis = image_composition(composed_im, composed_im_vis, patch_target, boxes[idx_topk[0]], is_square=False)
        # except:
        #     print('ssss')

        composed_dir = os.path.join(savedir, '6_single_car_composition')
        ensure_dir(composed_dir)
        save_path = composed_dir + '/' + filename
        imageio.imwrite(save_path, np.uint8(composed_im))
        
        composed_vis_dir = os.path.join(savedir, '6_single_car_composition_vis')
        ensure_dir(composed_vis_dir)
        save_path = composed_vis_dir + '/' + filename
        imageio.imwrite(save_path, np.uint8(composed_im_vis))

        txt_dir = os.path.join(savedir, 'one_sample_clip_score.txt')
        with open(txt_dir, 'a') as f:
            f.write(filename + ' ' + str(clip_picked[0]) + '\n')
        
        composed_dir = os.path.join(savedir, '6_single_car_composition_mask')
        ensure_dir(composed_dir)
        save_path = composed_dir + '/' + filename
        composed_mask = composed_mask[:, :, 0]
        imageio.imwrite(save_path, np.uint8(composed_mask))
        
        bbx_cpu = boxes[idx_topk[0]].detach().cpu().numpy()
        txt_dir = os.path.join(savedir, 'bbox_.txt')
        with open(txt_dir, 'a') as f:
            f.write(filename + ' ' + str(bbx_cpu[0]) + ' ' + str(bbx_cpu[1]) + ' ' + str(bbx_cpu[2]) + ' ' + str(bbx_cpu[3]) + '\n')


        
        ''' 4.3.7 Topk proposal car prediction + aspect ratio + CLIP, only clip '''
        num_query = 5
        clip_thr = clip[:, 0]
        logits_topk, idx_topk = torch.topk(clip_thr, num_query, dim=0, largest=False)

        composed_im = np.asarray(im).astype(int)
        composed_im_vis = composed_im
        try:        
            composed_im, composed_im_vis = image_composition(composed_im, composed_im_vis, patch_target, boxes[idx_topk[0]], is_square=False)
        except:
            print('ssss') 

        composed_dir = os.path.join(savedir, '7_single_car_composition')
        ensure_dir(composed_dir)
        save_path = composed_dir + '/' + filename
        imageio.imwrite(save_path, np.uint8(composed_im))
        
        composed_vis_dir = os.path.join(savedir, '7_single_car_composition_vis')
        ensure_dir(composed_vis_dir)
        save_path = composed_vis_dir + '/' + filename
        imageio.imwrite(save_path, np.uint8(composed_im_vis))

        txt_dir = os.path.join(savedir, 'one_sample_clip_score2.txt')
        with open(txt_dir, 'a') as f:
            f.write(filename + ' ' + str(logits_topk[0]) + '\n')
        

        ''' 4.3.8 Topk proposal car prediction + aspect ratio + CLIP, only probas '''
        num_query = 5
        logits_topk, idx_topk = torch.topk(probas_car, num_query, dim=0)

        composed_im = np.asarray(im).astype(int)
        composed_im_vis = composed_im
        try:        
            composed_im, composed_im_vis = image_composition(composed_im, composed_im_vis, patch_target, boxes[idx_topk[0]], is_square=False)
        except:
            print('ssss')

        composed_dir = os.path.join(savedir, '8_single_car_composition')
        ensure_dir(composed_dir)
        save_path = composed_dir + '/' + filename
        imageio.imwrite(save_path, np.uint8(composed_im))
        
        composed_vis_dir = os.path.join(savedir, '8_single_car_composition_vis')
        ensure_dir(composed_vis_dir)
        save_path = composed_vis_dir + '/' + filename
        imageio.imwrite(save_path, np.uint8(composed_im_vis))

        txt_dir = os.path.join(savedir, 'one_sample_clip_score3.txt')
        with open(txt_dir, 'a') as f:
            f.write(filename + ' ' + str(logits_topk[0]) + '\n')
        

        ''' 4.3.9 Topk proposal car prediction + aspect ratio + CLIP, CLIP + probas without real car '''
        num_query = 5
        keep_bbx_large = bbx_area > 0.1
        print(torch.sum(keep_bbx_large))
        # keep_bbx_large = torch.ge(bbx_area, 0.05)
        # logits_topk, idx_topk = torch.topk(clip[:, 0] * probas_car * ~keep_car, num_query, dim=0)
        # logits_topk, idx_topk = torch.topk((clip[:, 0] * probas_proposal[:, 0]) * ~keep_obj * keep_bbx_large, num_query, dim=0)
        # logits_topk, idx_topk = torch.topk((clip[:, 0] * probas[:, 0]), num_query, dim=0)
        # logits_topk, idx_topk = torch.topk((clip[:, 0] * probas_proposal[:, 0]), num_query, dim=0)
        # logits_topk, idx_topk = torch.topk((keep_car_ / clip[:, 0] * keep_area), num_query, dim=0)
        # logits_topk, idx_topk = torch.topk((probas_obj / clip[:, 0] * keep_area), num_query, dim=0)
        logits_topk, idx_topk = torch.topk((keep_car_ / clip[:, 0]), num_query, dim=0)


        # print(clip[:, 0])
        # sssss

        logits_pick = clip[idx_topk, 0]

        composed_im = np.asarray(im).astype(int)
        composed_im_vis = composed_im
        try:          
            composed_im, composed_im_vis, composed_mask = image_composition(composed_im, composed_im_vis, patch_target, boxes[idx_topk[0]], is_square=False, is_mask=True)
        except:
            print('ssss')

        # import pdb; pdb.set_trace()
        # composed_im, composed_im_vis, composed_mask = image_composition(composed_im, composed_im_vis, patch_target, boxes[idx_topk[0]], is_square=False, is_mask=True)

        composed_dir = os.path.join(savedir, '9_single_car_composition')
        ensure_dir(composed_dir)
        save_path = composed_dir + '/' + filename
        imageio.imwrite(save_path, np.uint8(composed_im))
        
        composed_vis_dir = os.path.join(savedir, '9_single_car_composition_vis')
        ensure_dir(composed_vis_dir)
        save_path = composed_vis_dir + '/' + filename
        imageio.imwrite(save_path, np.uint8(composed_im_vis))

        txt_dir = os.path.join(savedir, 'one_sample_clip_score4.txt')
        with open(txt_dir, 'a') as f:
            f.write(filename + ' ' + str(logits_pick[0]) + '\n')
        
        composed_dir = os.path.join(savedir, '9_single_car_composition_mask')
        ensure_dir(composed_dir)
        save_path = composed_dir + '/' + filename
        composed_mask = composed_mask[:, :, 0]
        imageio.imwrite(save_path, np.uint8(composed_mask))
        
        bbx_cpu = boxes[idx_topk[0]].detach().cpu().numpy()
        print(bbx_cpu)
        txt_dir = os.path.join(savedir, 'bbox.txt')
        with open(txt_dir, 'a') as f:
            f.write(filename + ' ' + str(bbx_cpu[0]) + ' ' + str(bbx_cpu[1]) + ' ' + str(bbx_cpu[2]) + ' ' + str(bbx_cpu[3]) + '\n')

        bbx_cpu = boxes[idx_topk[1]].detach().cpu().numpy()
        print(bbx_cpu)
        txt_dir = os.path.join(savedir, 'bbox1.txt')
        with open(txt_dir, 'a') as f:
            f.write(filename + ' ' + str(bbx_cpu[0]) + ' ' + str(bbx_cpu[1]) + ' ' + str(bbx_cpu[2]) + ' ' + str(bbx_cpu[3]) + '\n')

        bbx_cpu = boxes[idx_topk[2]].detach().cpu().numpy()
        print(bbx_cpu)
        txt_dir = os.path.join(savedir, 'bbox2.txt')
        with open(txt_dir, 'a') as f:
            f.write(filename + ' ' + str(bbx_cpu[0]) + ' ' + str(bbx_cpu[1]) + ' ' + str(bbx_cpu[2]) + ' ' + str(bbx_cpu[3]) + '\n')

        bbx_cpu = boxes[idx_topk[3]].detach().cpu().numpy()
        print(bbx_cpu)
        txt_dir = os.path.join(savedir, 'bbox3.txt')
        with open(txt_dir, 'a') as f:
            f.write(filename + ' ' + str(bbx_cpu[0]) + ' ' + str(bbx_cpu[1]) + ' ' + str(bbx_cpu[2]) + ' ' + str(bbx_cpu[3]) + '\n')

        bbx_cpu = boxes[idx_topk[4]].detach().cpu().numpy()
        print(bbx_cpu)
        txt_dir = os.path.join(savedir, 'bbox4.txt')
        with open(txt_dir, 'a') as f:
            f.write(filename + ' ' + str(bbx_cpu[0]) + ' ' + str(bbx_cpu[1]) + ' ' + str(bbx_cpu[2]) + ' ' + str(bbx_cpu[3]) + '\n')


        # GT location
        # txt_dir = os.path.join(savedir, 'bbox_gt.txt')
        # with open(txt_dir, 'a') as f:
        #     f.write(filename + ' ' + str(w1) + ' ' + str(h1) + ' ' + str(w2) + ' ' + str(h2) + '\n')


    elif is_recompose:

        ''' 4.5 Visualize: threshold on prediction on car '''
        # print(np.shape(keep_all_category))
        # print(np.shape(boxes))
        # ssss
        keep = keep_all_category[:, 0]
        # box_label = [id2name[int(item)] for item in labels[keep]]
        box_label = [str(round(item, 3)) for item in probas[:, 0][keep].cpu().detach().numpy()]
        pred_dict = {
            'boxes': boxes[keep],
            'size': torch.Tensor([im_inpaint_t.shape[1], im_inpaint_t.shape[2]]),
            'box_label': box_label,
            'image_id': 0
        }
        visual_dir = os.path.join(savedir, '1_visualize_objects_threshold_on_pred_car')
        ensure_dir(visual_dir)
        vslzr = COCOVisualizer()
        vslzr.savefig(im_inpaint_t, pred_dict, filename=filename, savedir=visual_dir, dpi=500)

        ''' 4.6 Visualize: threshold on prediction on pedestrian'''
        keep = keep_all_category[:, 1]
        # box_label = [id2name[int(item)] for item in labels[keep]]
        box_label = [str(round(item, 3)) for item in probas[:, 1][keep].cpu().detach().numpy()]
        print(np.shape(box_label))
        print(np.shape(keep))
        pred_dict = {
            'boxes': boxes[keep],
            'size': torch.Tensor([im_inpaint_t.shape[1], im_inpaint_t.shape[2]]),
            'box_label': box_label,
            'image_id': 0
        }
        visual_dir = os.path.join(savedir, '2_visualize_objects_threshold_on_pred_pedestrian')
        ensure_dir(visual_dir)
        vslzr = COCOVisualizer()
        vslzr.savefig(im_inpaint_t, pred_dict, filename=filename, savedir=visual_dir, dpi=500)

        ''' 4.7 Visualize: top-1 of each object for CLIP loss'''
        top = 1
        logits_top1, idx_top1 = torch.topk(clip, top, dim=0, largest=False)
        # box_label = [id2name[int(item)] for item in labels[keep]]
        print(idx_top1)
        box_label = [str(round(item, 3)) for item in clip[idx_top1, np.arange(patch_num)][0].cpu().detach().numpy()]
        pred_dict = {
            'boxes': boxes[idx_top1[0, :], :],
            'size': torch.Tensor([im_inpaint_t.shape[1], im_inpaint_t.shape[2]]),
            'box_label': box_label,
            'image_id': 0
        }
        visual_dir = os.path.join(savedir, '3_visualize_object_matching')
        ensure_dir(visual_dir)
        vslzr = COCOVisualizer()
        vslzr.savefig(im_inpaint_t, pred_dict, filename=filename, savedir=visual_dir, dpi=500)

        ''' 4.8 Visualize '''
        composed_im = np.asarray(im).astype(int)
        composed_im_vis = composed_im
        for i in range(patch_num):
            patch_name = patch_list[i][:-7]
            patch_class = class_map[patch_name]
            # clip_patch = clip[:, i] * keep_all_category[:, patch_class]
            # clip_patch = clip[:, i] * probas[:, patch_class] * keep_all_category[:, patch_class]
            # clip_patch = probas[:, patch_class] * keep_all_category[:, patch_class] - clip[:, i]
            clip_patch = probas[:, patch_class] * (1 - clip[:, i])

            top = 5
            logits_topk, idx_topk = torch.topk(clip_patch, top, dim=0)

            patch_i = imageio.imread(os.path.join(patch_root, filename, patch_list[i]))
            composed_im, composed_im_vis = image_composition(composed_im, composed_im_vis, patch_i, boxes[idx_topk[0]])
            clip[idx_topk, :] = 1

            # box_label = [str(round(item, 3)) for item in logits_top1.cpu().detach().numpy()]
            # pred_dict = {
            #     'boxes': boxes[idx_topk],
            #     'size': torch.Tensor([im_inpaint_t.shape[1], im_inpaint_t.shape[2]]),
            #     'box_label': box_label,
            #     'image_id': 0
            # }
        
        composed_dir = os.path.join(savedir, '4_composition_objects')
        ensure_dir(composed_dir)
        save_path = composed_dir + '/' + filename + '.png'
        imageio.imwrite(save_path, np.uint8(composed_im))

        composed_vis_dir = os.path.join(savedir, '5_composition_objects_vis')
        ensure_dir(composed_vis_dir)
        save_path = composed_vis_dir + '/' + filename + '.png'
        imageio.imwrite(save_path, np.uint8(composed_im_vis))
            

            






        num_query = 20
        # clip_thr = clip[:, -1] * keep
        # print(np.shape(clip_thr), np.shape(clip))
        # ssss
        # print(np.shape(keep))
        # print(np.shape(clip))
        # print(np.shape(probas))
        # ssss
        try: 
            # logits_im, idx_patch = clip.max(-1) # 200
            logits_im, idx_patch = clip.min(-1) # 200
            logits_topk, idx_topk = torch.topk(logits_im, num_query, dim=0, largest=False)
            patch_idx_ = idx_patch[idx_topk]
            if is_unique: 
                unique, idx, counts = torch.unique(patch_idx_, dim=0, sorted=True, return_inverse=True, return_counts=True)
                _, ind_sorted = torch.sort(idx, stable=True)
                cum_sum = counts.cumsum(0)
                cum_sum = torch.cat((torch.tensor([0]).cuda(), cum_sum[:-1]))
                first_indicies = ind_sorted[cum_sum].flip(0).cpu()
                unique = unique.flip(0).cpu()
                num_q = len(unique)
            else:
                num_q = num_query
            
            composed_im = np.asarray(im).astype(int)
            composed_im_vis = composed_im
            
            for i in range(num_q):
                if is_unique: 
                    i = first_indicies[i]
                bbx_i = boxes[idx_topk[i]]
                sim = logits_im[idx_topk[i]]
                # print(sim, torch.max(logits_im), torch.min(logits_im))
                index_i = patch_idx_[i]
                # print(patch_idx_)
                # print(patch_list)
                # ssss
                # ssssss
                # if is_one: 
                #     if index_i==0:
                #         patch_i_dir = os.path.join(patch_root, filename, patch_list[index_i])
                #         patch_i = imageio.imread(patch_i_dir)
                #         composed_im, composed_im_vis, composed_im_poisson = image_composition(composed_im, composed_im_vis, composed_im_poisson, patch_i, bbx_i)
                # elif index_i>patch_num_gt:
                #     patch_i_dir = os.path.join(patch_root, filename, patch_list[index_i])
                #     patch_i = imageio.imread(patch_i_dir)
                #     composed_im, composed_im_vis, composed_im_poisson = image_composition(composed_im, composed_im_vis, composed_im_poisson, patch_i, bbx_i)
                # patch_i_dir = os.path.join(patch_root, filename)
                patch_i_dir = os.path.join(patch_root, filename, patch_list[index_i])
                # print(patch_i_dir)
                # ssss
                patch_i = imageio.imread(patch_i_dir)
                composed_im, composed_im_vis = image_composition(composed_im, composed_im_vis, patch_i, bbx_i)

                composed_dir = os.path.join(savedir, 'composition_objects')
                # print(composed_dir)
                # print(filename)
                # ssss
                ensure_dir(composed_dir)
                save_path = composed_dir + '/' + filename + '.png'
                imageio.imwrite(save_path, np.uint8(composed_im))

                composed_vis_dir = os.path.join(savedir, 'composition_objects_vis')
                ensure_dir(composed_vis_dir)
                save_path = composed_vis_dir + '/' + filename + '.png'
                imageio.imwrite(save_path, np.uint8(composed_im_vis))
        except: 
            print('ssss')

f.close()

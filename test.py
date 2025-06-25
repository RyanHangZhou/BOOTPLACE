import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import imageio
from PIL import Image
import PIL
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO as AnnLoader

from models import build_model
from datasets import build_dataset
import util.misc as utils
from util.test_utils import ensure_dir, image_composition


def get_args_parser():
    parser = argparse.ArgumentParser('BOOTPLACE test script', add_help=False)

    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)

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
    parser.add_argument('--dataset_file', default='Cityscapes')
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--is_mask', action='store_true')
    parser.add_argument('--pretrained_model', type=str, help='path of pretrained model')
    parser.add_argument('--im_root', type=str, help='path of test set')

    return parser


def main(args):
    device = torch.device(args.device)

    print("Building model...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    print(f"Loading pretrained weights from {args.pretrained_model}")
    checkpoint = torch.load(args.pretrained_model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    def set_dropout_train(module):
        for m in module.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    model.eval()
    set_dropout_train(model.transformer)

    print("Building dataset...")
    dataset_val = build_dataset('test', args)
    data_loader_val = DataLoader(dataset_val,
                                 batch_size=args.batch_size,
                                 sampler=torch.utils.data.SequentialSampler(dataset_val),
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)

    print("Starting evaluation...")
    for samples, patches, targets in data_loader_val:
        samples = samples.to(device)
        patches = patches.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Load image metadata
        ann_file = Path(args.data_path) / 'test' / 'annotations.json'
        ann_data = AnnLoader(ann_file)
        image_id = ann_data.loadImgs(targets[0]["ids"].item())[0]["file_name"]

        patch_num = len(targets)
        outputs = model(samples, patches, targets)

        # Prediction scores
        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]
        probas = pred_logits.softmax(-1)[:, :-1]
        probas_obj = 1 - pred_logits.softmax(-1)[:, -1]
        probas_car = probas[:, 0]
        clip1 = outputs['pred_clip'][0, :, :patch_num]
        clip2 = outputs['pred_clip2'][0, :patch_num, :].T
        clip_score = (clip1.softmax(dim=0) + clip2.softmax(dim=0)) / 2

        # Select top-k car predictions
        num_query = 10
        _, idx_topk_car = torch.topk(probas_car, num_query, dim=0)
        keep_car = torch.zeros_like(probas_car, dtype=torch.bool)
        keep_car[idx_topk_car] = True

        # Rank top-k proposals based on inverse clip matching
        _, idx_topk = torch.topk((keep_car / clip_score[:, 0]), num_query, dim=0)

        # Read input background image
        img_h, img_w = 540, 960
        img_path = Path(args.data_path) / 'test' / 'backgrounds_single' / image_id
        im = Image.open(img_path).convert("RGB").resize((img_w, img_h))
        composed_im = np.asarray(im).astype(np.int32)
        composed_im_vis = composed_im.copy()

        # Read object patch
        patch_path = Path(args.data_path) / 'test' / 'objects_single' / image_id
        patch_target = imageio.imread(patch_path)

        try:
            # Compose object into background
            composed_im, composed_im_vis, composed_mask = image_composition(
                composed_im,
                composed_im_vis,
                patch_target,
                pred_boxes[idx_topk[0]],
                is_square=False,
                is_mask=True
            )
        except Exception as e:
            print(f"[Warning] Skipping composition for '{image_id}': {str(e)}")
            continue

        # Save composed output
        composed_dir = Path(args.output_dir) / 'composites'
        ensure_dir(composed_dir)
        save_path = composed_dir / image_id
        imageio.imwrite(str(save_path), composed_im.astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BOOTPLACE testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

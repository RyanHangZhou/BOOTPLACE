import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('BOOTPLACE Training Script', add_help=False)

    # Optimization parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # Model parameters
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=200, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # Loss parameters
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set_cost_class', default=0, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--ce_loss_coef', default=0, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--clip_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)

    # Dataset parameters
    parser.add_argument('--dataset_file', default='Cityscapes', type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--is_mask', action='store_true')
    parser.add_argument('--save_freq', default=20, type=int)

    return parser

def main(args):
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Dataset
    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('test', args)
    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=torch.utils.data.BatchSampler(
                                       torch.utils.data.RandomSampler(dataset_train),
                                       args.batch_size,
                                       drop_last=True),
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val,
                                 batch_size=args.batch_size,
                                 sampler=torch.utils.data.SequentialSampler(dataset_val),
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    # Resume
    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        for k in ["class_embed.weight", "class_embed.bias", "query_embed.weight"]:
            checkpoint["model"].pop(k, None)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        if not args.eval and all(k in checkpoint for k in ['optimizer', 'lr_scheduler', 'epoch']):
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args.data_path)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # Train
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        # Checkpoint
        if args.output_dir:
            ckpts = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_freq == 0:
                ckpts.append(output_dir / f'checkpoint{epoch:04}.pth')
            for p in ckpts:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, p)

        # Eval
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args.data_path)
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    for name in ['latest.pth', f'{epoch:03}.pth'] if epoch % args.save_freq == 0 else ['latest.pth']:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f'Training completed in {total_time}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('BOOTPLACE Training & Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

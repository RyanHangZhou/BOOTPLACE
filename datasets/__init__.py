import torch.utils.data
import torchvision

from .cityscapes import build as build_cityscapes
# from .opa import build as build_opa


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    return dataset.ann_data


def build_dataset(image_set, args):
    if args.dataset_file == 'Cityscapes':
        return build_cityscapes(image_set, args)
    # elif args.dataset_file == 'OPA':
    #     return build_opa(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

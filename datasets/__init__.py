import torch.utils.data
import torchvision

from .cityscapes import build as build_cityscapes
# from .opa import build as build_opa


def get_coco_api_from_dataset(dataset):
    """
    Traverse nested Subset wrappers to retrieve the COCO-style annotation API.
    
    Args:
        dataset (torch.utils.data.Dataset or Subset): Dataset possibly wrapped by Subset.

    Returns:
        COCO-style annotation API object (e.g., pycocotools.coco.COCO).
    """
    for _ in range(10):  # Limit depth to avoid infinite loop
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    return getattr(dataset, 'ann_data', None)


def build_dataset(split, args):
    """
    Factory method for dataset construction based on the specified configuration.

    Args:
        split (str): Dataset split, e.g., 'train' or 'test'.
        args (Namespace): Configuration object containing dataset type and paths.

    Returns:
        torch.utils.data.Dataset: Constructed dataset instance.
    """
    dataset_type = args.dataset_file.lower()

    if dataset_type == 'cityscapes':
        return build_cityscapes(split, args)
    # elif dataset_type == 'opa':
    #     return build_opa(split, args)

    raise ValueError(f"Unsupported dataset type: '{args.dataset_file}'. "
                     f"Available options: ['Cityscapes'/*OPA*/].")

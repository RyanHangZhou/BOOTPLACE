<h1 align="center">BOOTPLACE: Bootstrapped Object Placement <br> with Detection Transformers</h1>
<!-- <p align="center"><a href="https://arxiv.org/abs/2412.01506"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a> -->
<!-- <a href='https://trellis3d.github.io'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a> -->
<!-- </p> -->
<p align="center"><img src="assets/teaser.png" width="80%"></p>

<span style="font-size: 16px; font-weight: 600;">BOOTPLACE is a paradigm that formulates object placement as a placement-by-detection problem. It begins by identifying suitable regions of interest for object placement. This is achieved by training a specialized detection transformer on object-subtracted backgrounds, enhanced with multi-object supervisions. It then semantically associates each target compositing object with detected regions based on their complementary characteristics. Through a boostrapped training approach applied to randomly object-subtracted images, it enforces meaningful placements through extensive paired data augmentation. </span>


<!-- # BOOTPLACE
PyTorch implementation for paper BOOTPLACE: Bootstrapped Object Placement with Detection Transformers. -->


***Check out our [Project Page](https://ryanhangzhou.github.io/bootplace/) for more visual demos!***

<!-- Updates -->
## ⏩ Updates

**03/20/2025**
- Release training code and pretrained models.

**06/24/2025**
- Release inference code and data.

<!-- TODO List -->
<!-- ## 🚧 TODO List
- [x] Release training code
- [x] Release pretrained models
- [x] Release dataset
- [x] Release inference code -->



<!-- Installation -->
## 📦 Installation

### Prerequisites
- **System**: The code is currently tested only on **Linux**. 
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The code has been verified on NVIDIA A6000 GPUs.  
- **Software**:   
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.6 or higher is required. 

  Create a new conda environment named `BOOTPLACE` and install the dependencies: 
  ```
  conda env create --file=BOOTPLACE.yml
  ```
  Download DETR-R50 pretrained models for finetuning [here](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth) and put it in the directory ```weights/detr-r50-e632da11.pth```. 

  

<!-- Pretrained Models -->
## 🤖 Pretrained Models

We provide the following pretrained models:

| Model | Description | #Params | Download |
| --- | --- | --- | --- |
| BOOTPLACE_Cityscapes | Multiple supervision | 43.5M | [Download](https://drive.google.com/file/d/1ptWd_3NTTqkk1I4XI9GubRpgVopfnN1p/view?usp=sharing) |
| BOOTPLACE_OPA | Single supervision | Coming soon | Coming soon |


<!-- Usage -->
<!-- ## 💡 Usage

### Minimal Example

Here is an [example](test.py) of how to use the pretrained models for object placement.

 -->


<!-- Dataset -->
## 📚 Dataset
<!-- We use the data from Cityscapes and [OPA](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA). Note that we have provided instructions to prepare customized Cityscapes dataset for object composition in supplementary material. -->
<!-- We provide **TRELLIS-500K**, a large-scale dataset containing 500K 3D assets curated from [Objaverse(XL)](https://objaverse.allenai.org/), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [HSSD](https://huggingface.co/datasets/hssd/hssd-models), and [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k), filtered based on aesthetic scores. Please refer to the [dataset README](DATASET.md) for more details. -->
We provide a large-scale street-scene vehicle placement dataset curated from [Cityscapes](https://www.cityscapes-dataset.com/). 
The file structures are: 
├── train
    ├── backgrounds:
        ├── imgID.png
        ├── ……
    ├── objects:
        ├── imgID:
            ├── object_name_ID.png
            ├── ……
        ├── ……
    ├── location:
        ├── imgID:
            ├── object_name_ID.txt
            ├── ……
        ├── ……
    ├── annotations.json
├── test
    ├── backgrounds:
        ├── imgID.png
        ├── ……
    |── backgrounds_single
        ├── imgID.png
        ├── ……
    ├── objects:
        ├── imgID:
            ├── object_name_ID.png
            ├── ……
        ├── ……
    ├── objects_single:
        ├── imgID:
            ├── object_name_ID.png
            ├── ……
        ├── ……
    ├── location:
        ├── imgID:
            ├── object_name_ID.txt
            ├── ……
        ├── ……
    ├── location_single:
        ├── imgID:
            ├── object_name_ID.txt
            ├── ……
        ├── ……
    ├── annotations.json



## Training

To train a model on Cityscapes:
```
python -m main \
    --epochs 200 \
    --batch_size 2 \
    --save_freq 10 \
    --set_cost_class 1 \
    --ce_loss_coef 1 \
    --num_queries 120 \
    --eos_coef 0.1 \
    --lr 1e-4 \
    --data_path data/Cityscapes \
    --output_dir results/Cityscapes_ckpt\
    --resume weights/detr-r50-e632da11.pth
```

## Evaluation
```
python test.py \
    --num_queries 120 \
    --data_path data/Cityscapes \
    --pretrained_model 'results/Cityscapes_ckpt/checkpoint.pth' \
    --im_root 'data/Cityscapes/test' \
    --output_dir 'results/Cityscape_inference'
```




<!-- License -->
## ⚖️ License

This project is licensed under the terms of the MIT license.



<!-- Citation -->
## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{zhou2025bootplace,
    title   = {{BOOTPLACE}: Bootstrapped Object Placement with Detection Transformers},
    author  = {Zhou, Hang and Zuo, Xinxin and Ma, Rui and Cheng, Li},
    journal = {arXiv preprint arXiv:2503.21991}, 
    year    = {2025}
}
```


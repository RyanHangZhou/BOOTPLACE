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

**03/19/2025**
- Release training code.

<!-- TODO List -->
## 🚧 TODO List
- [x] Release training code
- [ ] Release pretrained models
- [ ] Release dataset
- [ ] Release inference code



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
  

<!-- Pretrained Models -->
## 🤖 Pretrained Models

We provide the following pretrained models:

| Model | Description | #Params | Download |
| --- | --- | --- | --- |
| BOOTPLACE_Cityscapes | Multiple supervision | 43.5M | [Download](https://huggingface.co/JeffreyXiang/TRELLIS-image-large) |
| BOOTPLACE_OPA | Single supervision | Coming Soon | Coming Soon |


<!-- Usage -->
## 💡 Usage

### Minimal Example

Here is an [example](test.py) of how to use the pretrained models for object placement.




## Requirements
- Python 3 with numpy, scikit-image, opencv-python, and PyTorch.


## Datasets and weights
We use the data from Cityscapes and [OPA](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA). Note that we have provided instructions to prepare customized Cityscapes dataset for object composition in supplementary material.
Please download the DETR pretrained model for finetuning [here](https://github.com/facebookresearch/detr).

We also provide the pre-trained network weights for Cityscapes and OPA.
(The weights will be provided here in a personal cloud storage. Omitted for anonymity.)


## Training

To train a model on Cityscapes:
```
python -m main \
    --epochs 200 \
    --batch_size 2\
    --save_freq 10 \
    --set_cost_class 1 \
    --ce_loss_coef 1 \
    --num_queries 120 \
    --eos_coef 0 \
    --lr 1e-4 \
    --data_path data/Cityscapes \
    --output_dir /results/data_Cityscapes\
    --resume weights/detr-r50-e632da11.pth
```
where ```--eos_coef``` is a hyperparameter to control the weight for non-object detection; ```--data_path``` specify the training data.


## Evaluation
```
python test_one_sample2_recomp.py \
    --is_recompose True \
    --num_queries 120 \
    --eos_coef 0.01 \
    --pretrained_model 'data_Cityscapes/checkpoint0059.pth' \
    --im_root 'data/Cityscapes/test' \
    --savedir '/results/data_Cityscapes_recomposition_0059'
```


<!-- Dataset -->
## 📚 Dataset

We provide **TRELLIS-500K**, a large-scale dataset containing 500K 3D assets curated from [Objaverse(XL)](https://objaverse.allenai.org/), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [HSSD](https://huggingface.co/datasets/hssd/hssd-models), and [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k), filtered based on aesthetic scores. Please refer to the [dataset README](DATASET.md) for more details.



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
    journal = {arXiv preprint arXiv:}, 
    year    = {2025}
}
```


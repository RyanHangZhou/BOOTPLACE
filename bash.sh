# 1. no smoothing
CUDA_VISIBLE_DEVICES=0 python -m main \
    --epochs 200 \
    --batch_size 2\
    --save_freq 10 \
    --set_cost_class 1 \
    --ce_loss_coef 1 \
    --num_queries 120 \
    --eos_coef 0.1 \
    --lr 1e-4 \
    --data_path data/Cityscapes \
    --output_dir results_redo/data_Cityscapes_no_smooth \
    --resume weights/detr-r50-e632da11.pth

CUDA_VISIBLE_DEVICES=0 python test_one_sample2_recomp.py \
    --is_recompose True \
    --num_queries 120 \
    --eos_coef 0.1 \
    --pretrained_model '/local-scratch2/hang/CPPP3/results_redo/data_Cityscapes_no_smooth/checkpoint0089.pth' \
    --im_root '/local-scratch2/hang/CPPP3/data/Cityscapes_repaint/test' \
    --savedir '/local-scratch2/hang/CPPP3/results_redo/data_Cityscapes_no_smooth_repaint0089'


# 1. no smoothing + eos = 0.01
CUDA_VISIBLE_DEVICES=0 python -m main \
    --epochs 200 \
    --batch_size 2\
    --save_freq 10 \
    --set_cost_class 1 \
    --ce_loss_coef 1 \
    --num_queries 120 \
    --eos_coef 0.01 \
    --lr 1e-4 \
    --data_path data/Cityscapes \
    --output_dir results_redo/data_Cityscapes_no_smooth_eos001 \
    --resume weights/detr-r50-e632da11.pth
CUDA_VISIBLE_DEVICES=0 python test_one_sample2_recomp.py \
    --is_recompose True \
    --num_queries 120 \
    --eos_coef 0.01 \
    --pretrained_model '/local-scratch2/hang/CPPP3/results_redo/data_Cityscapes_eos001/checkpoint0059.pth' \
    --im_root '/local-scratch2/hang/CPPP3/data/Cityscapes_repaint/test' \
    --savedir '/local-scratch2/hang/CPPP3/results_redo/data_Cityscapes_eos001_repaint0059'

# 1. no smoothing + eos = 0.00
CUDA_VISIBLE_DEVICES=0 python -m main \
    --epochs 200 \
    --batch_size 2\
    --save_freq 10 \
    --set_cost_class 1 \
    --ce_loss_coef 1 \
    --num_queries 120 \
    --eos_coef 0 \
    --lr 1e-4 \
    --data_path data/Cityscapes \
    --output_dir results_redo/data_Cityscapes_no_smooth_eos0 \
    --resume weights/detr-r50-e632da11.pth
CUDA_VISIBLE_DEVICES=0 python test_one_sample2_recomp.py \
    --is_recompose True \
    --num_queries 120 \
    --eos_coef 0.01 \
    --pretrained_model '/local-scratch2/hang/CPPP3/results_redo/data_Cityscapes_no_smooth_eos0/checkpoint0059.pth' \
    --im_root '/local-scratch2/hang/CPPP3/data/Cityscapes_repaint/test' \
    --savedir '/local-scratch2/hang/CPPP3/results_redo/data_Cityscapes_no_smooth_eos0_repaint0059'




# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


python -m main \
    --epochs 200 \
    --batch_size 2\
    --save_freq 10 \
    --set_cost_class 1 \
    --ce_loss_coef 1 \
    --num_queries 120 \
    --eos_coef 0 \
    --lr 1e-4 \
    --data_path /public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes \
    --output_dir results/data_Cityscapes\
    --resume weights/detr-r50-e632da11.pth

python test_one_sample_attention.py \
    --is_recompose True \
    --num_queries 120 \
    --eos_coef 0 \
    --pretrained_model 'results/data_Cityscapes/checkpoint0059.pth' \
    --im_root '/public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes/test' \
    --savedir 'results/data_Cityscapes/repaint0059_att'

python test_one_sample_one.py \
    --is_recompose False \
    --num_queries 120 \
    --eos_coef 0 \
    --pretrained_model 'results/data_Cityscapes/checkpoint0059.pth' \
    --im_root '/public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes_compose/test' \
    --savedir 'results/data_Cityscapes/repaint0039'


python test_one_sample2_new.py \
    --is_one True \
    --num_queries 120 \
    --pretrained_model 'results/data_Cityscapes/checkpoint0059.pth' \
    --im_root '/public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes/test' \
    --savedir 'results/data_Cityscapes_test_again'


python test_one_sample2_multiple.py \
    --is_recompose False \
    --num_queries 120 \
    --eos_coef 0.0 \
    --pretrained_model 'results/data_Cityscapes/checkpoint0059.pth' \
    --im_root '/public/zhouhang/composition/BOOTCOMP/data/Mapillary_Vistas' \
    --savedir 'results/data_mapv_recomposition_0059'


####################################
#GT-inference
# R1-Q2
python test_one_sample2_new2.py \
    --is_one True \
    --num_queries 120 \
    --pretrained_model 'results/data_Cityscapes/checkpoint0059.pth' \
    --im_root '/public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes/test' \
    --savedir 'results/data_Cityscapes_gt_test3'
python



##### R1-Q2
python -m main \
    --epochs 200 \
    --batch_size 2\
    --save_freq 10 \
    --set_cost_class 1 \
    --ce_loss_coef 1 \
    --num_queries 120 \
    --eos_coef 0 \
    --lr 1e-4 \
    --data_path /public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes \
    --output_dir results/data_Cityscapes_single_label\
    --resume weights/detr-r50-e632da11.pth


# python test_one_sample2_new2.py \
#     --is_one True \
#     --num_queries 120 \
#     --pretrained_model 'results/data_Cityscapes/checkpoint0059.pth' \
#     --im_root '/public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes/test' \
#     --savedir 'results/data_Cityscapes_test_again'

# R1-Q2
python test_one_sample2_new2.py \
    --is_one True \
    --num_queries 120 \
    --pretrained_model 'results/data_Cityscapes_single_label/checkpoint0059.pth' \
    --im_root '/public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes/test' \
    --savedir 'results/data_Cityscapes_single_label_test2'
python


python util/topk_iou_cpp3.py
python


#### R2-no scene cond
python -m main \
    --epochs 200 \
    --batch_size 2\
    --save_freq 10 \
    --set_cost_class 1 \
    --ce_loss_coef 1 \
    --num_queries 120 \
    --eos_coef 0 \
    --lr 1e-4 \
    --data_path /public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes \
    --output_dir results/data_Cityscapes_no_scene_cond\
    --resume weights/detr-r50-e632da11.pth
python


python test_one_sample2_new2.py \
    --is_one True \
    --num_queries 120 \
    --pretrained_model 'results/data_Cityscapes_no_scene_cond/checkpoint0059.pth' \
    --im_root '/public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes/test' \
    --savedir 'results/data_Cityscapes_no_scene_cond_test2'
python


python test_one_sample2_new2.py \
    --is_one True \
    --num_queries 120 \
    --pretrained_model 'results/data_Cityscapes/checkpoint0059.pth' \
    --im_root '/public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes/test' \
    --savedir 'results/data_Cityscapes_no_scene_cond_test3'
python

python test_one_sample2_new2.py \
    --is_one True \
    --num_queries 120 \
    --pretrained_model 'results/data_Cityscapes/checkpoint0059.pth' \
    --im_root '/public/zhouhang/composition/transfer/ARComp/data/cityscapes/Cityscapes/test' \
    --savedir 'results/data_Cityscapes_ablation_no_cond'
python



python test_one_sample2_new2_detr_pred.py \
    --is_one True \
    --num_queries 120 \
    --pretrained_model 'results/data_Cityscapes/checkpoint0059.pth' \
    --im_root '/data/hang/placement/ARComp/data/cityscapes/Cityscapes/test' \
    --savedir 'results/data_Cityscapes_ablation_detr_pred_t'
python






###############################
python -m main \
    --epochs 200 \
    --batch_size 2 \
    --save_freq 10 \
    --set_cost_class 1 \
    --ce_loss_coef 1 \
    --num_queries 120 \
    --eos_coef 0 \
    --lr 1e-4 \
    --data_path data/Cityscapes \
    --output_dir results/data_Cityscapes_n2\
    --resume weights/detr-r50-e632da11.pth



# python test.py \
#     --is_one True \
#     --num_queries 120 \
#     --pretrained_model 'results/data_Cityscapes_n/checkpoint0119.pth' \
#     --im_root 'data/Cityscapes/test' \
#     --savedir 'results/Cityscape_test_n'


python test.py \
    --is_one False \
    --num_queries 120 \
    --pretrained_model 'results/data_Cityscapes_n/checkpoint0059.pth' \
    --im_root 'data/Cityscapes/test' \
    --savedir 'results/Cityscape_test_n3'



print(root)
import pdb; pdb.set_trace()





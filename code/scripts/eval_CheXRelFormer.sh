#!/usr/bin/env bash

gpus=0
data_name=CXRData
net_G=CheXRelFormer #This is the best model
split=test
project_name=exp1_CheXRelFormer_CXRData_b16_lr0.00006_adamw_train_val_100_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256
checkpoints_root=~/CheXRelFormer/checkpoints/checkpoints_final
checkpoint_name=best_ckpt.pt
img_size=256
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)

CUDA_VISIBLE_DEVICES=1 python ~/SDPL/code/eval_CheXRelFormer.py --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size}  --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
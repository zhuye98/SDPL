#!/usr/bin/env bash

gpus=0
data_name=CXRData
net_G=SDPL #This is the best model
split=test
project_name=exp1_SDPL_CXRData_b16_lr0.00006_adamw_train_val_100_linear_ce
checkpoints_root=~/SDPL/checkpoints/checkpoints_final
checkpoint_name=best_ckpt.pt
img_size=256

CUDA_VISIBLE_DEVICES=1 python ~/SDPL/code/eval_SDPL.py --split ${split} --net_G ${net_G} --img_size ${img_size}  --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
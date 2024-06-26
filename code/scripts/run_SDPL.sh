#!/usr/bin/env bash
#GPUs
gpus=0

#Set paths
checkpoint_root=~/SDPL/checkpoints/checkpoints_final

data_name=CXRData

img_size=256
batch_size=16
lr=0.00006
     
max_epochs=100

# net_G:
# 
net_G=SDPL   #change this to the model you want to use

lr_policy=linear
optimizer=adamw  #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                     

#Train and Validation splits
split=train         #trainval
split_val=val      #test
project_name=exp1_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}

CUDA_VISIBLE_DEVICES=0 python ~/SDPL/code/main_SDPL.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} 
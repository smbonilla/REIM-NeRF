#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1"
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH 


echo "Training REIM-Nerf, our full model, which is a combination of NeRF conditioned on the light-source location and sparce depth supervision"

dataset_root_dir=/workspaces/REIM-NeRF/data/GP-processed/
width=320 # c3vd - 270
height=240 # c3vd - 216
total_samples=30000
rgb_loss=L2
depth_loss=L1
init=glorot
dataset_type=reim_json

model_type=ls_loc
depth_ratio=0.03


for dataset_path in ${dataset_root_dir}/*
do 

    dataset_name=$(basename "$dataset_path")
    # remove next line
    if [ $dataset_name == "sigmoid" ]; then
            
        
        image_count=$(ls -l ${dataset_path}/images | wc -l)
        epochs=$(($total_samples/image_count))
        step1=$((epochs/2))
        step2=$((((epochs-step1)/2)+step1))

        python -m scripts.train \
        --dataset_name ${dataset_type} \
        --root_dir ${dataset_path} \
        --N_importance 64 --img_wh ${width} ${height} \
        --num_epochs ${epochs} --batch_size 1024 \
        --optimizer adam --lr 5e-4 \
        --num_gpus 2 \
        --lr_scheduler steplr --decay_step ${step1} ${step2} --decay_gamma 0.5 \
        --init_type ${init} \
        --variant ${model_type} \
        --rgb_loss ${rgb_loss} \
        --supervise_depth \
        --depth_loss ${depth_loss} \
        --depth_loss_levels all \
        --depth_ratio ${depth_ratio} \
        --exp_name c3vd/main_results_iter${total_samples}_w${width}_h${height}/${model_type}_depth_${depth_ratio}/${dataset_name}
    fi
    # exp_name is {dataset_name}/{experiment_name}/{model_type}/{sub_dataset_name}
done
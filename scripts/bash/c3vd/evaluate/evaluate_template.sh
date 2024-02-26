#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

python -m scripts.evaluate_GP \
    --dataset_root_dir /workspaces/REIM-NeRF/data/GP-processed/ \
    --predictions_root_dir /workspaces/REIM-NeRF/results/reim_json_render/c3vd/main_results_iter_w320_h240/ls_loc_depth_0.03/ \
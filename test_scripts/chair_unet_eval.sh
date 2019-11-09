#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=3

python ../test_unet.py \
    --config_filepath "/home/apsk14/srn_segmentation/test_scripts/config_chairs_test.yml" \
    --srn_path "/media/data2/apsk14/models/chair_vanilla_srn/epoch_0650_iter_025000.pth" \
    --unet_path "/media/data2/apsk14/logging/Chairs/checkpoints/epoch_0022_iter_080000.pth" \
    --logging_root "/media/data2/apsk14/logging/final_results/Chairs/chair_unet_single" \

#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=0

python ../test.py \
    --config_filepath "/home/apsk14/srn_segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data2/apsk14/logging/Chair_unet_30shot/epoch_0499_iter_000500.pth" \
    --logging_root "/media/data2/apsk14/logging/new_results/Chairs/chair_unet30_multi" \
    --eval_mode 'unet' \

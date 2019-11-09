#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=2

python ../test.py \
    --config_filepath "/home/apsk14/srn_segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data2/apsk14/models/chair_seg_srn/epoch_0650_iter_025000.pth" \
    --logging_root "/media/data2/apsk14/logging/final_results/Chairs/chair_srn_single" \

#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=2

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data3/apsk14/srn_new_logging/Chair/unet46_multi/checkpoints/epoch_0057_iter_002000.pth" \
    --logging_root "/media/data3/apsk14/srn_new_results/Chair/chair_unet46_multi" \
    --eval_mode 'unet' \

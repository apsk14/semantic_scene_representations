#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=7

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_tables_test.yml" \
    --checkpoint_path "/media/data2/sitzmann/srn-segmentation/table_unet_30shot_dp0.4/checkpoints/epoch_0499_iter_000500.pth" \
    --logging_root "/media/data1/apsk14/table_unet30_multi" \
    --eval_mode 'unet' \

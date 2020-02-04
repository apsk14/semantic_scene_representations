#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=9

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data2/sitzmann/srn-segmentation/chair_linear_reverse_single/checkpoints/epoch_60000_iter_060000.pth" \
    --linear_path "/media/data2/sitzmann/srn-segmentation/Chairs_linear/checkpoints/epoch_2000_iter_020000.pth" \
    --logging_root "/media/data1/apsk14/chair_linear_reverse_single_new" \
    --eval_mode 'linear' \
    --max_num_instances 10 \

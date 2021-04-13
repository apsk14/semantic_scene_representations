#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=1

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data3/apsk14/srn_new_logging/Chair/reverse_single/checkpoints/epoch_24999_iter_025000.pth" \
    --logging_root "/media/data3/apsk14/chair_reverse_single" \
    --eval_mode 'srn' \
    --max_num_instances 10 \
    #--linear_path "/media/data2/sitzmann/srn-segmentation/Chairs_linear/checkpoints/epoch_2000_iter_020000.pth" \
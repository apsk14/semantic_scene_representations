#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=2

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_tables_test.yml" \
    --checkpoint_path "/media/data2/sitzmann/srn-segmentation/tables_vanilla_latent_single/checkpoints/epoch_0859_iter_065000.pth" \
    --linear_path "/media/data2/sitzmann/srn-segmentation/table_linear/epoch_4000_iter_040000.pth" \
    --logging_root "/media/data1/apsk14/table_linear_single" \
    --eval_mode 'linear' \


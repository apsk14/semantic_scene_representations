#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=9

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_tables_test.yml" \
    --checkpoint_path "/media/data2/sitzmann/srn-segmentation/tables_class_latent_single/checkpoints/epoch_1005_iter_080000.pth" \
    --logging_root "/media/data1/apsk14/table_srn_single" \
    --eval_mode 'srn'

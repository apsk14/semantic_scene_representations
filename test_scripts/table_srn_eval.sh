#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=4

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_tables_test.yml" \
    --checkpoint_path "/media/data3/apsk14/srn_new_logging/Table/latent_seg_multi/checkpoints/epoch_0019_iter_080000.pth" \
    --logging_root "/media/data3/apsk14/srn_new_results/Table/table_srn_multi" \
    --eval_mode 'srn'

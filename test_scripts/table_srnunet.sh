#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=6

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_tables_test.yml" \
    --checkpoint_path "/media/data3/apsk14/srn_new_logging/Table/latent_vanilla_single/checkpoints/epoch_1005_iter_080000.pth" \
    --unet_path "/media/data3/apsk14/srn_new_logging/Table/srnunet/srnunet_10/checkpoints/epoch_0500_iter_001000.pth" \
    --logging_root "/media/data3/apsk14/srn_new_results/Table/table_srnunet10_single" \
    --eval_mode 'unet' \

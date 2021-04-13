#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=7

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_single/checkpoints/epoch_1370_iter_080000.pth" \
    --unet_path "/media/data3/apsk14/srn_new_logging/Chair/srnunet/srnunet_all/checkpoints/epoch_0009_iter_002000.pth" \
    --logging_root "/media/data3/apsk14/srn_new_results/Chair/srnunet/chair_srnunet1214_single" \
    --eval_mode 'unet' \

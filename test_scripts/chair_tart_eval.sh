#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=8

python ../test_tatarchenko.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --tatarchenko_ckpt "/media/data3/apsk14/srn_new_logging/Chair/tart_train/checkpoints/epoch_0025_iter_045000.pth" \
    --linear_ckpt "/media/data3/apsk14/srn_new_logging/Chair/tart_linear/checkpoints/epoch_5000_iter_005000.pth" \
    --logging_root "/media/data3/apsk14/srn_new_results/Chair/chair_tatarchenko_linear" \

#/media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_single/checkpoints/epoch_1370_iter_080000.pth
#media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_multi/checkpoints/epoch_0026_iter_080000.pth
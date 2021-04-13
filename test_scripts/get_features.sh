#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=1

python ../get_features.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data3/apsk14/srn_new_logging/Chair/train_vanilla/checkpoints/epoch_0010_iter_105000.pth" \
    --linear_path "/media/data3/apsk14/srn_new_logging/Chair/linear_single_random/checkpoints/epoch_20000_iter_040000.pth" \
    --logging_root "/media/data1/apsk14/tsne" \
    --eval_mode 'linear' \
    --specific_observation_idcs 1
#1000
#/media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_single/checkpoints/epoch_1370_iter_080000.pth
#media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_multi/checkpoints/epoch_0026_iter_080000.pth
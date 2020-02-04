#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=8

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data2/sitzmann/srn-segmentation/chairs_vanilla_latent_multi/checkpoints/epoch_0022_iter_065000.pth" \
    --linear_path "/media/data2/sitzmann/srn-segmentation/chair_mlp_single/checkpoints/epoch_0500_iter_001000.pth" \
    --logging_root "/media/data1/apsk14/chair_mlp_multi" \
    --eval_mode 'linear' \


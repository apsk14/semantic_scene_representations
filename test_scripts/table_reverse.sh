#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=8

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data2/sitzmann/srn-segmentation/chairs_latent_single/checkpoints/epoch_1384_iter_080000.pth" \
    --logging_root "/media/data1/apsk14/chair_srn_single" \
    --eval_mode 'srn'

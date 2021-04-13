#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=6

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data3/apsk14/srn_new_logging/Chair/latent_seg_multi/checkpoints/epoch_0026_iter_080000.pth" \
    --logging_root "/media/data3/apsk14/srn_new_results/Chair/chair_srn_multi" \
    --eval_mode 'srn'

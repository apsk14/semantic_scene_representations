#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=3

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_single/checkpoints/epoch_1370_iter_080000.pth" \
    --linear_path "/media/data3/apsk14/srn_new_logging/Chair/one_v_all_2/checkpoints/epoch_5000_iter_010000.pth" \
    --logging_root "/home/apsk14/data/one_v_all_2" \
    --eval_mode 'linear' \

#/media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_single/checkpoints/epoch_1370_iter_080000.pth
#media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_multi/checkpoints/epoch_0026_iter_080000.pth
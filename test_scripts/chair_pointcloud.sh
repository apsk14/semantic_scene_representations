#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=2

python ../test.py \
    --config_filepath "/home/apsk14/srn-segmentation/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_single/checkpoints/epoch_1370_iter_080000.pth" \
    --linear_path "/media/data3/apsk14/srn_new_logging/Chair/linear_single/checkpoints/epoch_12500_iter_025000.pth" \
    --logging_root "/home/apsk14/data/chair_linear_pointcloud" \
    --eval_mode 'linear' \
    --max_num_instances 4 \
    --point_cloud

#/media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_single/checkpoints/epoch_1370_iter_080000.pth
#media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_multi/checkpoints/epoch_0026_iter_080000.pth
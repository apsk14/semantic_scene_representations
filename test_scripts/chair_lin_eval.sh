#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=0

python ../test.py \
    --config_filepath "/home/amit/projects/semantic_scene_representations/test_scripts/config_chairs_test.yml" \
    --checkpoint_path "/media/hugespace/amit/srn_new_logging/Chair/latent_vanilla/checkpoints/epoch_9999_iter_050000.pth" \
    --logging_root "/media/hugespace/amit/srn_new_logging/Chair/results/linear_single_new" \
    --linear_path "/media/hugespace/amit/srn_new_logging/Chair/linear_30/checkpoints/epoch_0642_iter_004500.pth" \
    --eval_mode 'linear' \
    --max_num_instances 5  \
    --batch_size 32 \
    --input_idcs 65 \
    --point_cloud

#/media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_single/checkpoints/epoch_1370_iter_080000.pth
#media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_multi/checkpoints/epoch_0026_iter_080000.pth
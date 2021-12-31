#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=0

python ../test.py \
    --config_filepath "/home/amit/projects/semantic_scene_representations/test_scripts/config_chairs_test.yml" \ 
    --checkpoint_path "/media/hugespace/amit/srn_new_logging/Chair/latent_vanilla/checkpoints/epoch_9999_iter_010000.pth" \
    --logging_root "/media/hugespace/amit/srn_new_logging/Chair/Chair/vanilla_single" \
    --eval_mode 'srn' \
    --max_num_instances 10 \

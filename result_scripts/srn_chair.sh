#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=0

python ../test.py \
    --config_filepath ~/projects/semantic_scene_representations/config_test_chair.yml \
    --checkpoint_path "/media/hugespace/amit/pretrained_models/test_vanilla_50/checkpoints/epoch_0026_iter_080000.pth" \
    --log_dir srn_30shot \
    --eval_mode 'srn' \
    --max_num_instances 10  \
    --input_idcs 65 \
    --point_cloud
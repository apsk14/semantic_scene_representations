#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=0

python ../test.py \
    --config_filepath ~/projects/semantic_scene_representations/config_test_table.yml \
    --checkpoint_path "/media/hugespace/amit/srn_logging/Table/test_vanilla_1/checkpoints/epoch_0000_iter_000000.pth"  \
    --log_dir srn_30shot \
    --eval_mode 'srn' \
    --max_num_instances 10  \
    --batch_size 1 \
    --input_idcs 65 \
    --point_cloud

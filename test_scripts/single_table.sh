#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python ../train.py \
--config_filepath ~/projects/semantic_scene_representations/config_test_table.yml \
--log_dir test_vanilla_1 \
--checkpoint_path /media/hugespace/amit/srn_logging/Table/vanilla_srn/checkpoints/epoch_0000_iter_000000.pth \
--overwrite_embeddings \
--img_sidelengths 128 --max_steps_per_img_sidelength 10000 --batch_size_per_img_sidelength 4 \
--specific_observation_idcs 65 \
--class_weight=0. \
--max_num_instances_train 10 \


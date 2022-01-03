#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python ../train.py  \
--config_filepath~/projects/semantic_scene_representations/config_train_table.yml \
--log_dir train_vanilla \
--max_num_instances_train 100 \
--img_sidelengths 128 \
--batch_size_per_img_sidelength 4 \
--max_steps_per_img_sidelength 20000 \
--steps_til_ckpt 10000 \
--class_weight=0.


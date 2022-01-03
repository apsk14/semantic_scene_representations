#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0 
python ../train.py  \
--config_filepath "config_chair.yml" \
--log_dir train_semantic \
--max_num_instances_train 10 \
--img_sidelengths 128 \
--batch_size_per_img_sidelength 4 \
--max_steps_per_img_sidelength 100000 \
--steps_til_ckpt 10000 \

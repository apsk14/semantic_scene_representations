#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0 
python ../train.py  \
--config_filepath "path to config_train_table.yml" \
--log_dir "put desired name of logging dir" \
--max_num_instances_train 20 \
--img_sidelengths 64,128 \
--batch_size_per_img_sidelength 4,8 \
--max_steps_per_img_sidelength 5000,150000 \
--steps_til_ckpt 10000 \

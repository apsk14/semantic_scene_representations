#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="put GPU here"
python ../train.py  \
--config_filepath "path to config_train_chair.yml" \
--log_dir "put desired name of logging dir" \
--img_sidelengths 64,128 \
--batch_size_per_img_sidelength 4,8 \
--max_steps_per_img_sidelength 5000,150000 \
--class_weight=0.


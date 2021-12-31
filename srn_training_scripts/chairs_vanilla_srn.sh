#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES='0' python ../train.py  \
--data_root /media/hugespace/amit/semantic_srn_data/Chair/Chair.train \
--obj_name 'Chair' \
--logging_root /media/hugespace/amit/srn_new_logging/Chair/train_vanilla \
--max_num_instances_train 100 \
--img_sidelengths 128 \
--batch_size_per_img_sidelength 4 \
--max_steps_per_img_sidelength 100000 \
--steps_til_ckpt 10000 \
--class_weight=0. --no_validation


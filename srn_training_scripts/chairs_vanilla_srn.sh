#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python ../train.py  \
--data_root /media/hugespace/amit/semantic_srn_data/Chair.train \
--obj_name 'Chair' \
--logging_root /media/hugespace/amit/srn_new_logging/Chair/train_vanilla \
--max_num_instances 50 \
--class_weight=0. --no_validation


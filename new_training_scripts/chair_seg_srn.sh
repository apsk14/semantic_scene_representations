#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=8 python ../train.py  \
--data_root /home/apsk14/data/final_data/Chair/Chair.train  \
--stat_root /media/data1/apsk14/srn_seg_data/Chair/Chair.train \
--obj_name 'Chair' \
--logging_root /media/data3/apsk14/srn_new_logging/Chair/train_seg \
--no_validation

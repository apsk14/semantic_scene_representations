#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 python ../train.py  \
--data_root /home/apsk14/data/final_data/Table/Table.train  \
--stat_root /media/data1/apsk14/srn_seg_data/Table/Table.train \
--obj_name 'Table' \
--logging_root /media/data3/apsk14/srn_new_logging/Table/train_seg \
--no_validation

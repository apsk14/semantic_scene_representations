#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python ../train.py --data_root /home/apsk14/data/final_data/Chair/Chair.train --logging_root /home/sitzmann/data/srn-segmentation/chairs_train_vanilla --class_weight=0. --no_validation

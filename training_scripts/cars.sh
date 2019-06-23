#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python ../run_srns.py --train_test train \
                      --data_root /home/sitzmann/data/deep_space/data/shapenet_cars_no_transp/ \
                      --logging_root /home/sitzmann/data/deep_space/logging/ \
                      --img_sidelength 32 \
                      --batch_size 128 \
		      --max_num_instances_train 50 \
		      --no_preloading \
		      --no_validation

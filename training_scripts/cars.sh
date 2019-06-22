#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7
#export CUDA_VISIBLE_DEVICES=9

python ../run_srns.py --train_test train \
                      --data_root /home/sitzmann/data/deep_space/data/shapenet_cars_no_transp/ \
                      --logging_root /home/sitzmann/data/deep_space/logging/ \
                      --img_sidelength 64 \
                      --batch_size 128 \
                      --lr 5e-5 \
                      --reg_weight 1.e-3 \
                      --no_validation \
                      --steps_til_ckpt 5000 \

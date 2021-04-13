#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python ../train_tatarchenko.py --config_filepath=./config_chairs_train.yml

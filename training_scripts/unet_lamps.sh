#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=1

python ../run_unet.py \
    --config_filepath "/home/apsk14/srn_segmentation/config_lamps_unet.yml"\







#python ../run_srn_unet.py --train_test train \
#                      --data_root /home/apsk14/data/final_data/Table/Table.train/ \
#                      --val_root /home/apsk14/data/final_data/Table/Table.train_val/ \
#                      --logging_root /home/sitzmann/data/deep_space/logging/srn_runs_final/Table/unet \
#                      --batch_size 128 \
#                      --max_epoch 500 \
#              --no_preloading \

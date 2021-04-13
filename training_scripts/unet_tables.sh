#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=2

python ../run_unet.py \
    --config_filepath "/home/apsk14/srn-segmentation/config_tables_unet.yml"\
    --logging_root "/media/data3/apsk14/srn_new_logging/Table/unet46_multi" \
    --steps_til_ckpt 1000 \
    --steps_til_val 1000 \
    --max_num_instances_val 50 \
    --max_num_instances_train 46 \





#python ../run_srn_unet.py --train_test train \
#                      --data_root /home/apsk14/data/final_data/Table/Table.train/ \
#                      --val_root /home/apsk14/data/final_data/Table/Table.train_val/ \
#                      --logging_root /home/sitzmann/data/deep_space/logging/srn_runs_final/Table/unet \
#                      --batch_size 128 \
#                      --max_epoch 500 \
#              --no_preloading \

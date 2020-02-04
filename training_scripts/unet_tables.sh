#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=8

python ../run_unet.py \
    --config_filepath "/home/apsk14/srn-segmentation/config_tables_unet.yml"\
    --max_num_instances_train 10 \
    --specific_observation_idcs 1,2,3 \
    --logging_root "/media/data2/sitzmann/srn-segmentation/table_unet_30shot_dp0.4" \
    --steps_til_ckpt 500 \
    --steps_til_val 100 \





#python ../run_srn_unet.py --train_test train \
#                      --data_root /home/apsk14/data/final_data/Table/Table.train/ \
#                      --val_root /home/apsk14/data/final_data/Table/Table.train_val/ \
#                      --logging_root /home/sitzmann/data/deep_space/logging/srn_runs_final/Table/unet \
#                      --batch_size 128 \
#                      --max_epoch 500 \
#              --no_preloading \

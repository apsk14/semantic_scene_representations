#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4
#/home/apsk14/data/final_data/Chair/Chair.val/
#--img_sidelength 128 \
#specific_observation_idcs 102 \
#

#Train
#python ../run_srns.py --train_test train \
#                      --data_root /home/apsk14/data/final_data/Table/Table.train/ \
#                      --logging_root /home/sitzmann/data/deep_space/logging/srn_runs_final/Table/train_vanilla \
#                      --batch_size 16 \
#                      --max_epoch 20 \
#              --no_preloading \
#		      --no_validation

##Optimize Latent


python ../train_linear.py \
    --config_filepath "/home/apsk14/srn-segmentation/config_tables_linear.yml" \
    --specific_observation_idcs 1,2,3 \
    --model_type linear \
    --logging_root /media/data3/apsk14/srn_new_logging/Table/linear_single \
    --specific_ins 161e0ae498eb2b9be3ea982be8e701e5 11f2882ca78cd85c9c75eb4326997fae 1950a6b5594160d39453d57bf9ae92b2 1d90363feb72fada9cdecade71f5dca2 376a1d212ab8971125f61c02205f9a5c 390e0db80fe12ef65fa6da97b9eb4a2f 7c1bcea89b0037a2d67bd369ec608dad 954f39bdb27c54cbeedb4c8fd29e2d1 c0e8eca9811baaf1237b12b19575e7ae ff127b5ab0e36faec3bec646284d5a6a \
    --no_validation \


#    --max_num_instances_train 10 \
#python ../run_srn_linear.py --train_test train \
#                      --data_root /home/apsk14/data/final_data/Chair/Chair.train/ \
#                      --logging_root /home/sitzmann/data/deep_space/logging/srn_runs_final/Chairs/10_shot \
#                      --batch_size 64 \
#                      --max_epoch 5000 \
#                      --max_num_instances_train 10 \
#                      --checkpoint /home/sitzmann/data/deep_space/logging/srn_runs_final/Chair/train_vanilla/logs/10_16/03-14-47_/epoch_0016_iter_225000.pth \
#                      --overwrite_embeddings \
#		      --no_preloading \
#		      --no_validation



#Point Cloud Only

#python ../run_srns_pc.py --train_test train \
#                      --data_root /home/apsk14/data/final_data/Chair/Chair.train/ \
#                      --logging_root /home/sitzmann/data/deep_space/logging/pc_runs \
#                      --batch_size 16 \
#                      --max_epoch 20 \
#		      --no_preloading \
#		      --no_validation


#python ../run_srns_pc.py --train_test train \
#                      --data_root /home/apsk14/data/final_data/Chair/Chair.val/ \
#                      --logging_root /home/sitzmann/data/deep_space/logging/pc_runs \
#                      --batch_size 16 \
#                      --max_epoch 20 \
#                      --checkpoint /home/sitzmann/data/deep_space/logging/pc_runs/logs/09_22/16-59-47_/epoch_0004_iter_070000.pth \
#                      --overwrite_embeddings \
#		      --no_preloading \
#		      --no_validation
















#/home/sitzmann/data/deep_space/logging/runs_amit/logs/09_10/12-02-56_

#/home/sitzmann/data/deep_space/logging/chkpt/08_25/15-38-59_/epoch_0016_iter_090000.pth
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
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


python ../train_srnunet.py \
    --config_filepath "/home/apsk14/srn-segmentation/config_chairs_srnunet.yml" \
    --specific_observation_idcs 1,2,3 \
    --logging_root /media/data3/apsk14/srn_new_logging/Chair/srnunet/srnunet_24/ \
    --steps_til_ckpt 1000 \
    --steps_til_val 1000 \
    --max_num_instances_val 50 \
    --max_num_instances_train 24 \
    #--specific_ins 1bcd9c3fe6c9087e593ebeeedbff73b 7d59399c37925cd7b1b9d9bf8d5ee54d 320b6f3ae2893d3c9f5d3fd8c90b27d2 92cae2e67b788eaa9dcc460592c0e125 470d626518cc53a2ff085529822a7226 b4c73f4772efcf69742728b30848ed03 60622d74c0712934a5817f81a1efa3cc dfc85456795d943bbadc820495ddb59 7a712ca74183d8c235836c728d324152 eb3029393f6e60713ae92e362c52d19d \
    #--max_num_instances_train 121 \

#    --max_num_instances_train 10 \
#    --specific_observation_idcs 1,2,3 \



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
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=8
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
python ../run_srn_linear.py --train_test train \
                      --data_root /home/apsk14/data/final_data/Chair/Chair.train/ \
                      --logging_root /home/sitzmann/data/deep_space/logging/srn_runs_final/Chairs/10_shot \
                      --batch_size 64 \
                      --max_epoch 5000 \
                      --max_num_instances_train 10 \
                      --checkpoint /home/sitzmann/data/deep_space/logging/srn_runs_final/Chair/train_vanilla/logs/10_16/03-14-47_/epoch_0016_iter_225000.pth \
                      --overwrite_embeddings \
		      --no_preloading \
		      --no_validation



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
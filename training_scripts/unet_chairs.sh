#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5
#/home/apsk14/data/final_data/Chair/Chair.val/
#--img_sidelength 128 \

#Train
python ../run_srn_unet.py --train_test train \
                      --data_root /home/apsk14/data/final_data/Table/Table.train/ \
                      --val_root /home/apsk14/data/final_data/Table/Table.train_val/ \
                      --logging_root /home/sitzmann/data/deep_space/logging/srn_runs_final/Table/unet \
                      --batch_size 128 \
                      --max_epoch 500 \
              --no_preloading \

##Optimize Latent
#python ../run_srns.py --train_test train \
#                      --data_root /home/apsk14/data/final_data/Chair/Chair.val/ \
#                      --logging_root /home/sitzmann/data/deep_space/logging/srn_runs_final/Chairs/latent_single \
#                      --batch_size 16 \
#                      --max_epoch 60 \
#                      --specific_observation_idcs 102 \
#                      --checkpoint /home/sitzmann/data/deep_space/logging/srn_runs_final/Chairs/train/logs/10_13/11-22-09_/epoch_0008_iter_125000.pth \
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
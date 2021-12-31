#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python ../train.py \
--data_root /media/hugespace/amit/semantic_srn_data/Chair/Chair.test \
--logging_root /media/hugespace/amit/srn_new_logging/Chair/latent_vanilla --no_validation \
--checkpoint_path /media/hugespace/amit/srn_new_logging/Chair/train_vanilla/checkpoints/epoch_0199_iter_010000.pth \
--overwrite_embeddings \
--obj_name Chair \
--img_sidelengths 64 --max_steps_per_img_sidelength 10000 --batch_size_per_img_sidelength 10 \
--specific_observation_idcs 65 \
--class_weight=0. \
--max_num_instances_train 15 \

##!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=8
#python ../train.py \
#--stat_root /media/data1/apsk14/srn_seg_data/Chair/Chair.test \
#--data_root /home/apsk14/data/final_data/Chair/Chair.test \
#--logging_root /media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_single --no_validation \
#--checkpoint_path /media/data3/apsk14/srn_new_logging/Chair/train_vanilla/checkpoints/epoch_0010_iter_105000.pth \
#--overwrite_embeddings \
#--obj_name Chair \
#--img_sidelengths 64,128 --max_steps_per_img_sidelength 5000,100000 --batch_size_per_img_sidelength 92,16 \
#--specific_observation_idcs 65 \
#--class_weight=0. \

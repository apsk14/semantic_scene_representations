#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python ../train.py \
--data_root /media/hugespace/amit/semantic_srn_data/Chair/Chair.test \
--logging_root /media/hugespace/amit/srn_new_logging/Chair/latent_vanilla --no_validation \
--obj_name Chair \
--checkpoint_path /media/hugespace/amit/srn_new_logging/Chair/train_vanilla/checkpoints/epoch_0079_iter_100000.pth \
--overwrite_embeddings \
--img_sidelengths 128 --max_steps_per_img_sidelength 50000 --batch_size_per_img_sidelength 4 \
--specific_observation_idcs 65 \
--class_weight=0. \
--max_num_instances_train 20 







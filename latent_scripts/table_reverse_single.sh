#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5
python ../train.py \
--data_root /home/apsk14/data/final_data/Table/Table.test \
--logging_root /media/data3/apsk14/srn_new_logging/Table/reverse_single --no_validation \
--stat_root /media/data1/apsk14/srn_seg_data/Table/Table.test \
--checkpoint_path /media/data3/apsk14/srn_new_logging/Table/reverse_single/checkpoints/epoch_14999_iter_015000.pth  \
--obj_name Table \
--start_step 15000 \
--img_sidelengths 128 --max_steps_per_img_sidelength 100000 --batch_size_per_img_sidelength 10 \
--specific_observation_idcs 65 \
--l1_weight=0. \
--max_num_instances_train 10 \
#--checkpoint_path /media/data3/apsk14/srn_new_logging/Table/train_seg/checkpoints/epoch_0008_iter_105000.pth \

#--overwrite_embeddings \
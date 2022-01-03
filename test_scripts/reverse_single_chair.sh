#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python ../update.py \
--config_filepath ~/projects/semantic_scene_representations/config_test_chair.yml \
--log_dir test_reverse_1 \
--checkpoint_path /media/hugespace/amit/srn_new_logging/Chair/train_vanilla/checkpoints/epoch_0079_iter_100000.pth \
--linear_path /media/hugespace/amit/srn_logging/Chair/update_linear_30/checkpoints/epoch_0714_iter_005000.pth \
--overwrite_embeddings \
--img_sidelengths 128 --max_steps_per_img_sidelength 10000 --batch_size_per_img_sidelength 4 \
--specific_observation_idcs 65 \
--max_num_instances_train 20 \

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

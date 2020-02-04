#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=9
python ../train.py \
--data_root /home/apsk14/data/final_data/Chair/Chair.test \
--stat_root /media/data1/apsk14/srn_seg_data/Chair/Chair.test \
--logging_root /media/data3/apsk14/srn_new_logging/Chair/latent_seg_multi --no_validation \
--checkpoint_path /media/data3/apsk14/srn_new_logging/Chair/train_seg/checkpoints/epoch_0010_iter_105000.pth \
--overwrite_embeddings \
--img_sidelengths 64,128 --max_steps_per_img_sidelength 5000,100000 --batch_size_per_img_sidelength 92,16 \
--obj_name Chair \
--class_weight=0.\

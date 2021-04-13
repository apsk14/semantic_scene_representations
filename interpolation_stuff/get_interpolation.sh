#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
python ../interpolate.py \
--data_root /home/apsk14/data/final_data/Chair/Chair.test \
--stat_root /media/data1/apsk14/srn_seg_data/Chair/Chair.test \
--logging_root /home/apsk14/data/chair_interpolation \
--checkpoint_linear /media/data3/apsk14/srn_new_logging/Chair/linear_single/checkpoints/epoch_10000_iter_020000.pth \
--checkpoint_srn  /media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_multi/checkpoints/epoch_0026_iter_080000.pth \
--obj_name Chair \
--embedding_size 256 \




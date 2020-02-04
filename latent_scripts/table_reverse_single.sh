#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5
python ../train.py \
--data_root /home/apsk14/data/final_data/Table/Table.test \
--logging_root /media/data2/sitzmann/srn-segmentation/tables_reverse --no_validation \
--stat_root /media/data1/apsk14/srn_seg_data/Table/Table.test \
--obj_name Table \
--checkpoint_path /media/data2/sitzmann/srn-segmentation/tables_train/checkpoints/epoch_0008_iter_105000.pth \
--overwrite_embeddings \
--img_sidelengths 64,128 --max_steps_per_img_sidelength 1000,100000 --batch_size_per_img_sidelength 10,10 \
--specific_observation_idcs 102 \
--l1_weight=0. \
--max_num_instances_train 10 \

#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=9
python ../train.py \
--data_root /home/apsk14/data/final_data/Chair/Chair.test \
--logging_root /media/data2/sitzmann/srn-segmentation/chairs_vanilla_latent_single --no_validation \
--checkpoint_path /media/data2/sitzmann/srn-segmentation/chairs_train_vanilla_continued/checkpoints/epoch_0005_iter_105000.pth \
--overwrite_embeddings \
--img_sidelengths 64,128 --max_steps_per_img_sidelength 5000,100000 --batch_size_per_img_sidelength 92,16 \
--specific_observation_idcs 102 \
--class_weight=0. \

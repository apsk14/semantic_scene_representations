#!/usr/bin/env bash

# reverse means that segmentation maps are observed at test time. 
export CUDA_VISIBLE_DEVICES=0
python ../train.py \
--config_filepath "path to config_test_chair.yml" \
--log_dir "" \
--checkpoint_path "path to trained semantic srn (ex: train_seg)" \
--overwrite_embeddings \
--img_sidelengths 128 --max_steps_per_img_sidelength 80000 --batch_size_per_img_sidelength 4 \
--specific_observation_idcs 65 \
--l1_weight=0. \

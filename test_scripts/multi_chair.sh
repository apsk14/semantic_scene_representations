#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python ../train.py \
--config_filepath "path to config_test_chair.yml" \
--log_dir "" \
--checkpoint_path "path to trained srn (ex: train_vanilla or train_seg)" \
--overwrite_embeddings \
--img_sidelengths 128 --max_steps_per_img_sidelength 80000 --batch_size_per_img_sidelength 4 \
--class_weight=0. \


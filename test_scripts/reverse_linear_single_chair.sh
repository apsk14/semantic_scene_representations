#!/usr/bin/env bash

#reverse here means that the test time observations are semantic maps instead of rgb images. This version uses linear updates.
export CUDA_VISIBLE_DEVICES=0
python ../update.py \
--config_filepath "path to config_test_chair.yml" \
--log_dir "" \
--checkpoint_path "path to trained srn (ex: train_vanilla)" \
--linear_path "path to linear regressor (ex: update_linear_30)" \
--overwrite_embeddings \
--img_sidelengths 128 --max_steps_per_img_sidelength 80000 --batch_size_per_img_sidelength 4 \
--specific_observation_idcs 65 \


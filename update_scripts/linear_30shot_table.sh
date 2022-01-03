#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python ../update.py \
    --config_filepath ~/projects/semantic_scene_representations/config_train_table.yml \
    --model_type linear \
    --specific_observation_idcs 0,1,2 \
    --log_dir update_linear_30_test \
    --checkpoint_path /media/hugespace/amit/srn_new_logging/Chair/train_vanilla/checkpoints/epoch_0079_iter_100000.pth \
    --img_sidelengths 128 \
    --batch_size_per_img_sidelength 4 \
    --max_steps_per_img_sidelength 5000 \
    --steps_til_ckpt 500 \
    --max_num_instances_train 10 \
    # instances used in the paper: 161e0ae498eb2b9be3ea982be8e701e5 11f2882ca78cd85c9c75eb4326997fae 1950a6b5594160d39453d57bf9ae92b2 1d90363feb72fada9cdecade71f5dca2 376a1d212ab8971125f61c02205f9a5c 390e0db80fe12ef65fa6da97b9eb4a2f 7c1bcea89b0037a2d67bd369ec608dad 954f39bdb27c54cbeedb4c8fd29e2d1 c0e8eca9811baaf1237b12b19575e7ae ff127b5ab0e36faec3bec646284d5a6a





#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python ../train_linear.py \
    --config_filepath "/home/amit/projects/semantic_scene_representations/config_main.yml" \
    --model_type linear \
    --specific_observation_idcs 0,1,2 \
    --logging_root /media/hugespace/amit/srn_new_logging/linear_30 \
    --checkpoint_path /media/hugespace/amit/srn_new_logging/Chair/train_vanilla/checkpoints/epoch_0079_iter_100000.pth \
    --no_validation \
    --img_sidelengths 128 \
    --batch_size_per_img_sidelength 4 \
    --max_steps_per_img_sidelength 5000 \
    --steps_til_ckpt 500 \
    --max_num_instances_train 10 \
    #--specific_ins 1bcd9c3fe6c9087e593ebeeedbff73b 7d59399c37925cd7b1b9d9bf8d5ee54d 320b6f3ae2893d3c9f5d3fd8c90b27d2 92cae2e67b788eaa9dcc460592c0e125 470d626518cc53a2ff085529822a7226 b4c73f4772efcf69742728b30848ed03 60622d74c0712934a5817f81a1efa3cc dfc85456795d943bbadc820495ddb59 7a712ca74183d8c235836c728d324152 eb3029393f6e60713ae92e362c52d19d

#--specific_ins 1eb2e372a204a61153baab6c8235f5db \
#    --max_num_instances_train 10 \
#    --specific_observation_idcs 1,2,3 \




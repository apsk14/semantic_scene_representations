#!/usr/bin/env bash


#Train
export CUDA_VISIBLE_DEVICES=0

python ../test.py \
    --config_filepath ~/projects/semantic_scene_representations/config_test_table.yml \
    --checkpoint_path "/media/hugespace/amit/srn_new_logging/Chair/latent_vanilla/checkpoints/epoch_9999_iter_050000.pth" \
    --log_dir linear_update_srn_30shot \
    --linear_path "/media/hugespace/amit/srn_logging/Chair/update_linear_30/checkpoints/epoch_0714_iter_005000.pth" \
    --eval_mode 'linear' \
    --max_num_instances 5  \
    --batch_size 32 \
    --input_idcs 65 \
    --point_cloud

#/media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_single/checkpoints/epoch_1370_iter_080000.pth
#media/data3/apsk14/srn_new_logging/Chair/latent_vanilla_multi/checkpoints/epoch_0026_iter_080000.pth
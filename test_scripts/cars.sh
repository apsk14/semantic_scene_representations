#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=9

python ../run_deepspace.py --train_test test \
                           --data_root /home/sitzmann/data/deep_space/data/shapenet_cars_no_transp_test/ \
                           --logging_root /home/sitzmann/data/deep_space/logging/ \
                           --use_images \
                           --img_sidelength 128 \
                           --mode hyper \
                           --tracing_steps 10 \
                           --implicit_nf 256 \
                           --embedding_size 256 \
                           --batch_size 16 \
                           --experiment_name cars_hypernet_non_affine_layernorm \
                           --num_objects 2433 \
                           --num_objects 50 \
                           --no_gan \
                           --reg_weight 1. \
                           --no_validation \
                           --no_preloading \
                           --checkpoint ~/data/deep_space/logging/logs/04_19/08-30-25_data_root_shapenet_cars_no_transp_val_root_None_max_steps_None_tracing_steps_10_experiment_name_cars_hypernet_non_affine_layernorm_lr_5e-05_gan_weight_0.0_l1_weight_200_kl_weight_1_proxy_weight_0_reg_/epoch_0003_iter_030000.pth


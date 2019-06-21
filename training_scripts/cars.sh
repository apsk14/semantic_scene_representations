#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7
#export CUDA_VISIBLE_DEVICES=9

#python ../run_srns.py --train_test train \
#                           --data_root /home/sitzmann/data/deep_space/data/shapenet_cars_no_transp/ \
#                           --logging_root /home/sitzmann/data/deep_space/logging/ \
#                           --use_images \
#                           --img_sidelength 64 \
#                           --mode hyper \
#                           --implicit_nf 256 \
#                           --embedding_size 256 \
#                           --batch_size 128 \
#                           --tracing_steps 5 \
#                           --experiment_name cars_hypernet_non_affine_layernorm \
#                           --no_gan \
#                           --lr 5e-5 \
#                           --reg_weight 1.e-3 \
#                           --no_validation \
#                           --steps_til_ckpt 5000 \
#                           --num_images 50 \
#                           --renderer fc

#python ../run_srns.py --train_test train \
#                           --data_root /home/sitzmann/data/deep_space/data/shapenet_cars_no_transp/ \
#                           --logging_root /home/sitzmann/data/deep_space/logging/ \
#                           --use_images \
#                           --img_sidelength 128 \
#                           --mode hyper \
#                           --implicit_nf 256 \
#                           --embedding_size 256 \
#                           --batch_size 16 \
#                           --tracing_steps 10 \
#                           --experiment_name cars_hypernet_non_affine_layernorm \
#                           --no_gan \
#                           --lr 5e-5 \
#                           --reg_weight 1.e-3 \
#                           --no_validation \
#                           --steps_til_ckpt 5000 \
#                           --no_preloading \
#                           --num_images 50 \
#                           --renderer fc \
#                           --checkpoint /home/sitzmann/data/deep_space/logging/logs/04_19/05-07-57_data_root_shapenet_cars_no_transp_val_root_None_max_steps_None_tracing_steps_5_experiment_name_cars_hypernet_non_affine_layernorm_lr_5e-05_gan_weight_0.0_l1_weight_200_kl_weight_1_proxy_weight_0_reg_w/epoch_0005_iter_005000.pth



python ../run_deepspace.py --train_test train \
                           --data_root /home/sitzmann/data/deep_space/data/shapenet_cars_no_transp/ \
                           --logging_root /home/sitzmann/data/deep_space/logging/ \
                           --use_images \
                           --img_sidelength 128 \
                           --mode hyper \
                           --implicit_nf 256 \
                           --embedding_size 256 \
                           --batch_size 16 \
                           --tracing_steps 10 \
                           --experiment_name cars_hypernet_non_affine_layernorm \
                           --no_gan \
                           --lr 5e-5 \
                           --reg_weight 1.e-3 \
                           --no_validation \
                           --steps_til_ckpt 5000 \
                           --no_preloading \
                           --num_images 50 \
                           --renderer fc \
                           --checkpoint /home/sitzmann/data/deep_space/logging/logs/04_19/05-07-57_data_root_shapenet_cars_no_transp_val_root_None_max_steps_None_tracing_steps_5_experiment_name_cars_hypernet_non_affine_layernorm_lr_5e-05_gan_weight_0.0_l1_weight_200_kl_weight_1_proxy_weight_0_reg_w/epoch_0005_iter_005000.pth \

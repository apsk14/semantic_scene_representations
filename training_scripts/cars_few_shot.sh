#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

python ../run_deepspace.py --train_test train \
                           --data_root /home/sitzmann/data/deep_space/data/shapenet_cars_no_transp_test/ \
                           --logging_root /home/sitzmann/data/deep_space/logging/few_shot \
                           --img_sidelength 128 \
                           --batch_size 12 \
                           --tracing_steps 10 \
                           --lr 5e-5 \
                           --reg_weight 1.e-3 \
                           --no_validation \
                           --steps_til_ckpt 5000 \
                           --no_preloading \
                           --num_images 3 \
                           --num_objects 50 \
                           --renderer fc \
                           --freeze_rendering \
                           --freeze_var \
                           --checkpoint  ~/data/deep_space/logging/logs/04_19/08-30-25_data_root_shapenet_cars_no_transp_val_root_None_max_steps_None_tracing_steps_10_experiment_name_cars_hypernet_non_affine_layernorm_lr_5e-05_gan_weight_0.0_l1_weight_200_kl_weight_1_proxy_weight_0_reg_/epoch_0024_iter_190000.pth \
                           --overwrite_embeddings




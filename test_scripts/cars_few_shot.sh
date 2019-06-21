#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

python ../run_deepspace.py --train_test test \
                           --data_root /home/sitzmann/data/deep_space/data/shapenet_cars_no_transp_test/ \
                           --logging_root /home/sitzmann/data/deep_space/logging/few_shot \
                           --use_images \
                           --img_sidelength 128 \
                           --mode hyper \
                           --implicit_nf 256 \
                           --embedding_size 256 \
                           --tracing_steps 10 \
                           --experiment_name cars_few_shot \
                           --no_gan \
                           --reg_weight 1.e-3 \
                           --no_preloading \
                           --num_objects 50 \
                           --renderer fc \
                           --checkpoint  ~/data/deep_space/logging/few_shot/logs/04_24/04-06-04_data_root_shapenet_cars_no_transp_test_val_root_None_max_steps_None_tracing_steps_10_experiment_name_cars_few_shot_lr_5e-05_gan_weight_0.0_l1_weight_200_kl_weight_1_proxy_weight_0_reg_weight_0.001_ste/epoch_1500_iter_018012.pth




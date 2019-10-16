 #!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7

#python ../run_deepspace.py --train_test test \
#                           --data_root /home/sitzmann/data/deep_space/data/shapenet_cars_no_transp_test/ \
#                           --logging_root /home/sitzmann/data/deep_space/logging/ \
#                           --use_images \
#                           --img_sidelength 128 \
#                           --mode hyper \
#                           --tracing_steps 10 \
#                           --implicit_nf 256 \
#                           --embedding_size 256 \
#                           --batch_size 16 \
#                           --experiment_name cars_hypernet_non_affine_layernorm \
#                           --num_objects 2433 \
#                           --num_objects 50 \
#                           --no_gan \
#                           --reg_weight 1. \
#                           --no_validation \
#                           --no_preloading \
#                           --checkpoint ~/data/deep_space/logging/logs/04_19/08-30-25_data_root_shapenet_cars_no_transp_val_root_None_max_steps_None_tracing_steps_10_experiment_name_cars_hypernet_non_affine_layernorm_lr_5e-05_gan_weight_0.0_l1_weight_200_kl_weight_1_proxy_weight_0_reg_/epoch_0003_iter_030000.pth
#

#python ../run_srns.py --train_test test \
#                      --data_root /home/apsk14/data/final_data/Chair/Chair.val/ \
#                      --logging_root /home/sitzmann/data/deep_space/logging/runs_amit/ \
#                      --img_sidelength 64 \
#                      --batch_size 3 \
#                      --max_epoch 100000 \
#                      --checkpoint /home/sitzmann/data/deep_space/logging/final_models/08_27/20-04-10_/epoch_84999_iter_085000.pth \
#		      --max_num_instances_train 3 \
#		      --specific_observation_idcs 0,1,2,3,4,5,6,7,8,9 \
#		      --no_preloading \
#		      --no_validation

python ../run_srns.py --train_test test \
                      --data_root /home/apsk14/data/final_data/Chair/Chair.val/ \
                      --logging_root /home/sitzmann/data/deep_space/logging/runs_amit/ \
                      --batch_size 1 \
                      --max_epoch 1 \
                      --checkpoint /home/sitzmann/data/deep_space/logging/runs_img_only/logs/10_11/20-34-14_/epoch_0012_iter_050000.pth \
		      --no_preloading \
		      --no_validation


#python ../run_srns_pc.py --train_test test \
#                      --data_root /home/apsk14/data/final_data/Chair/Chair.val/ \
#                      --logging_root /home/sitzmann/data/deep_space/logging/pc_runs/ \
#                      --batch_size 1 \
#                      --max_epoch 1 \
#                      --checkpoint /home/sitzmann/data/deep_space/logging/pc_runs/logs/09_23/08-06-55_/epoch_0002_iter_005000.pth \
#		      --no_preloading \
#		      --no_validation


#
# /home/sitzmann/data/deep_space/logging/final_models/09_02/14-24-30_/epoch_0039_iter_029360.pth
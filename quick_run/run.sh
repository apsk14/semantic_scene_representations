#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python ../test.py \
    --data_root /media/hugespace/amit/semantic_srn_data/Chair/Chair.test/ \
    --logging_root results/ \
    --obj_name Chair \
    --checkpoint_path "test_vanilla_50/epoch_0026_iter_080000.pth" \
    --linear_path "update_linear_30/epoch_2500_iter_005000.pth" \
    --eval_mode 'linear' \
    --max_num_instances 5  \
    --input_idcs 65 \
    --point_cloud

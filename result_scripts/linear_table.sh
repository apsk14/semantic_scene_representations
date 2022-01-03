#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python ../test.py \
    --config_filepath "path to config_test_table.yml" \
    --checkpoint_path "path to a test time srn (ex: test_vanilla_1 for single shot or test_vanilla_50 for 50 shot)" \
    --log_dir "" \
    --linear_path "path to linear regressor (ex: update_linear_30)" \
    --eval_mode 'linear' \
    --input_idcs 65 \
    --point_cloud
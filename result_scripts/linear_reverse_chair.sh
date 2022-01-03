#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python ../test.py \
    --config_filepath "path to config_test_chair.yml" \
    --checkpoint_path "path to test time reverse vanilla srn (ex: no pretrained models)" \
    --log_dir "" \
    --linear_path "path to linear regressor (ex: update_linear_30)" \
    --eval_mode 'linear' \
    --reverse \
    --input_idcs 65 \
    --point_cloud
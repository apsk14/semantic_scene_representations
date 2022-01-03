#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python ../test.py \
    --config_filepath "path to config_test_chair.yml" \
    --checkpoint_path "path to test time reverse semantic srn (ex: test_reverse_1)" \
    --log_dir "" \
    --eval_mode 'srn' \
    --input_idcs 65 \
    --point_cloud \
    --reverse
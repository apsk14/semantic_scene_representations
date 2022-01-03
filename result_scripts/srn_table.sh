#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python ../test.py \
    --config_filepath "path to config_test_table.yml" \
    --checkpoint_path "path to test time srn (ex: test_semantic_1 for single-shot semantic srn or test_vanilla_1 for single-shot vanilla srn" \
    --log_dir "" \
    --eval_mode 'srn' \
    --input_idcs 65 \
    --point_cloud

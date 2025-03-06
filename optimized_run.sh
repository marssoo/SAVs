python3 -m src.optimized_run \
    --model_name llava_ov_7b\
    --num_head 20\
    --quantize 4\
    --data_name natural_ret \
    --train_path data/naturalbench_ret_train.jsonl \
    --val_path data/naturalbench_ret_test_truncated20.jsonl\
    --file_path runs/nat_quant.json

python3 -m src.optimized_run \
    --model_name llava_ov_0.5b\
    --num_head 20\
    --data_name natural_ret \
    --train_path data/naturalbench_ret_train_truncated10.jsonl \
    --val_path data/naturalbench_ret_test_truncated5.jsonl\
    --file_path runs/nat_trunc.json

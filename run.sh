python3 -m src.run \
    --model_name llava_ov_7b \
    --quantize 8\
    --num_head 20\
    --data_name natural_ret \
    --train_path data/naturalbench_ret_train.jsonl \
    --val_path data/naturalbench_ret_test_truncated5.jsonl

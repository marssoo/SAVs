python3 -m src.save_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name natural_ret \
    --train_path comp_data/naturalbench_ret_train.jsonl \
    --val_path data/naturalbench_ret_test_truncated20.jsonl\
    --file_path runs/nat_ret_20_heads.json
    

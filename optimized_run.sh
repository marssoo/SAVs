python3 -m src.optimized_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name compare \
    --train_path comp_data/comp_train.jsonl \
    --val_path comp_data/comp_val_small.jsonl\
    --file_path runs/comp_20_heads.json

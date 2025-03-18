python3 -m src.run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name vlguard \
    --train_path truncated_data/vlguard_train_truncated5.json \
    --val_path truncated_data/vlguard_train_truncated5.json

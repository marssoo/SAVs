python3 -m src.optimized_run \
    --model_name llava_ov_0.5b\
    --num_head 20\
    --data_name MHalu \
    --train_path data/MHalubench_train.json \
    --val_path data/truncated/MHalubench_test_truncated10.json\
    --file_path runs/halu.json

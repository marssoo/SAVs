python3 -m src.run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name MHalu \
    --train_path data/truncated/MHalubench_test_truncated10.json \
    --val_path data/truncated/MHalubench_test_truncated10.json

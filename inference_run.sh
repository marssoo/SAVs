python3 -m src.inference \
    --model_name llava_ov_0.5b \
    --data_name natural_ret \
    --data_path data/naturalbench_ret_test_truncated5.jsonl\
    --saved_heads_path runs/nat_trunc.json\
    --output_path results/nat_trunc_results.json

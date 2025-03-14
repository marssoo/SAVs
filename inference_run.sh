python3 -m src.inference \
    --model_name llava_ov_0.5b \
    --data_name natural_ret \
    --data_path data/naturalbench_ret_test_truncated20.jsonl \
    --saved_heads_path runs/nat_ret_20_heads.json\
    --output_path results/natret_20_heads_is_similar.json

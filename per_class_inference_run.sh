python3 -m src.per_class_inference \
    --model_name llava_ov_0.5b \
    --data_name natural_ret \
    --data_path data/naturalbench_ret_test_truncated20.jsonl\
    --saved_heads_path runs/per_class_natret_20_heads.json\
    --output_path results/per_class_natret_20_heads_results.json

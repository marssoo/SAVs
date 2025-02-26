python3 -m src.inference \
    --model_name llava_ov_0.5b \
    --data_name natural_ret \
    --data_path data/naturalbench_ret_test_truncated20.jsonl \
    --saved_heads_path /home/onyxia/work/SAVs/runs/natural_bench_small_top_heads.json\
    --output_path results/20_heads_small_nat.json

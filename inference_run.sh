python3 -m src.inference \
    --model_name llava_ov_0.5b \
    --data_name compare \
    --data_path fma_data/fma_test.jsonl \
    --saved_heads_path runs/05b_fma_25_heads.json\
    --output_path results/05b_fma_25_heads_results.json

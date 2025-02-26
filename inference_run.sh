python3 -m src.inference \
    --model_name llava_ov_0.5b \
    --data_name compare \
    --data_path comp_data/comp_val.jsonl \
    --saved_heads_path runs/comp_20_heads.json\
    --output_path results/comp_20_heads_results.json

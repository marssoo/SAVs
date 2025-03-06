python3 -m src.inference \
    --model_name llava_ov_7b \
    --quantize 4\
    --data_name natural_ret \
    --data_path data/naturalbench_ret_test.jsonl\
    --saved_heads_path runs/nat_quant.json\
    --output_path results/nat_quant_results.json

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name pets \
    --train_path data/pets_train.json \
    --val_path data/pets_test.json\
    --utils base

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name MHalu \
    --train_path data/MHalubench_train.json \
    --val_path data/MHalubench_test.json\
    --utils base

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name vlguard \
    --train_path data/vlguard_train.json \
    --val_path data/vlguard_test.json\
    --utils base

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name eurosat \
    --train_path data/eurosat_train.json \
    --val_path data/eurosat_test.json \
    --utils base

python3 -m src.run_natural_ret \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name natural_ret \
    --train_path data/naturalbench_ret_train.jsonl \
    --val_path data/naturalbench_ret_test.jsonl \
    --utils base

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name pets \
    --train_path data/pets_train.json \
    --val_path data/pets_test.json\
    --utils artanh

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name MHalu \
    --train_path data/MHalubench_train.json \
    --val_path data/MHalubench_test.json\
    --utils artanh

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name vlguard \
    --train_path data/vlguard_train.json \
    --val_path data/vlguard_test.json\
    --utils artanh

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name eurosat \
    --train_path data/eurosat_train.json \
    --val_path data/eurosat_test.json \
    --utils artanh

python3 -m src.run_natural_ret \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name natural_ret \
    --train_path data/naturalbench_ret_train.jsonl \
    --val_path data/naturalbench_ret_test.jsonl \
    --utils artanh

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name pets \
    --train_path data/pets_train.json \
    --val_path data/pets_test.json\
    --utils polar

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name MHalu \
    --train_path data/MHalubench_train.json \
    --val_path data/MHalubench_test.json\
    --utils polar

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name vlguard \
    --train_path data/vlguard_train.json \
    --val_path data/vlguard_test.json\
    --utils polar

python3 -m src.flexible_run \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name eurosat \
    --train_path data/eurosat_train.json \
    --val_path data/eurosat_test.json \
    --utils polar

python3 -m src.run_natural_ret \
    --model_name llava_ov_0.5b \
    --num_head 20\
    --data_name natural_ret \
    --train_path data/naturalbench_ret_train.jsonl \
    --val_path data/naturalbench_ret_test.jsonl \
    --utils polar
python3 -m src.run_save_all \
    --model_name llava_ov_0.5b \
    --data_name natural_ret \
    --train_path data/naturalbench_ret_train.jsonl \
    --val_path data/naturalbench_ret_test.jsonl \

python3 -m src.run_save_all \
    --model_name llava_ov_0.5b \
    --data_name pets \
    --train_path data/pets_train.json \
    --val_path data/pets_test.json\
    --utils base

python3 -m src.run_save_all \
    --model_name llava_ov_0.5b \
    --data_name MHalu \
    --train_path data/MHalubench_train.json \
    --val_path data/MHalubench_test.json\
    --utils base

python3 -m src.run_save_all \
    --model_name llava_ov_0.5b \
    --data_name vlguard \
    --train_path data/vlguard_train.json \
    --val_path data/vlguard_test.json\
    --utils base

python3 -m src.run_save_all \
    --model_name llava_ov_0.5b \
    --data_name eurosat \
    --train_path data/eurosat_train.json \
    --val_path data/eurosat_test.json \
    --utils base
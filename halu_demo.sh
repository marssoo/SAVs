#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 -m src.run \
    --model_name llava_ov \
    --data_name pets \
    --train_path data/pets_train.json \
    --val_path data/pets_train.json

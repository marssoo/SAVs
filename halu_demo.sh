#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 -m run \
    --model_name llava_ov \
    --data_name MHalu \
    --train_path placeholder \
    --val_path placeholder 

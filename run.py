from utils import *
from model import *
from preprocess import *
from tqdm import tqdm
import torch
import argparse
torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error() 



def eval_reinforce(args):
    model = load_model(args.model_name, args.data_name)

    train_data = open_data(args.data_name, args.train_path)
    test_data = open_data(args.data_name, args.val_path)


    multimodal_embed = mllm_encode(model, train_data, num_head=20)

    correct_count = 0
    ###Checking how well it can classify a given query
    for item in tqdm(test_data):
        cur_class = mllm_classify(item, model, multimodal_embed)
        if item['label'] == cur_class:
            correct_count += 1
    print("Accuracy:", correct_count / len(test_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava_ov")
    parser.add_argument("--data_name", type=str, default="Mhalu")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    
    args = parser.parse_args()
    eval_reinforce(args)

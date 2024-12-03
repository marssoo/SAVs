from utils import *
from models import *
from preprocess import *
from tqdm import tqdm


import torch
import argparse
torch.set_grad_enabled(False)
from transformers.utils import logging
import sys
logging.set_verbosity_error() 


train_hallucinate_path = ""
train_nonhallucinate_path = ""
val_path = ""

use_knn = False
is_layer=False
num_head = 20

def eval_reinforce(args):
    model_helper = load_model(args.model_name, args.data_name)


    model_helper.mode_of_eval = args.cur_mode
    common_heads = torch.load("")
    

    train_hallucinate = open_data(args.data_name, train_hallucinate_path)
    train_nonhallucinate = open_data(args.data_name, train_nonhallucinate_path)
    val_data = open_data(args.data_name, val_path)


    ###Get the mean activations for each head and each class
    hallucinate_activations = get_sample_activations(train_hallucinate, model_helper, common_heads, use_knn, is_layer=is_layer)
    nonhallucinate_activations = get_sample_activations(train_nonhallucinate, model_helper, common_heads, use_knn, is_layer=is_layer)
    combined_activations = torch.vstack((hallucinate_activations, nonhallucinate_activations))

    success_count = [0 for _ in range(nonhallucinate_activations.shape[1])]
    ###Reuse the training set
    for item in tqdm(train_hallucinate + train_nonhallucinate):

        cur_activations = get_query_activations([item], model_helper, common_heads, is_layer=is_layer).squeeze(dim=0)
        if item["claim_label"] == "hallucination":
            label = 0
        else:
            label = 1

        ###Record how each head's mean activation perform when classifying the current input
        record_head_performance(combined_activations, cur_activations, label, success_count, use_knn)

    arr = np.array(success_count)
    # Pick the top-20 heads that has highest classification accuracy
    k = num_head
    topk_indices = np.argsort(arr)[-k:][::-1]

    new_heads = []
    for item in topk_indices.tolist():

        new_heads.append(common_heads[item])

    ###Recalculate the activation with new heads.
    hallucinate_activations = get_sample_activations(train_hallucinate, model_helper, new_heads, is_layer=is_layer, is_knn=use_knn)
    nonhallucinate_activations = get_sample_activations(train_nonhallucinate, model_helper, new_heads, is_layer=is_layer, is_knn=use_knn)
    combined_activations = torch.vstack((hallucinate_activations, nonhallucinate_activations))


    correct_count = 0
    ###Checking how well it can classify a test set
    for item in tqdm(val_data):
        cur_activations = get_query_activations([item], model_helper, new_heads, is_layer=is_layer).squeeze(dim=0)
        top_k_examples = retrieve_examples(combined_activations, cur_activations, use_knn)
        if top_k_examples[0] == 0:
            cur_ans = "hallucination"
        else:
            cur_ans = "non-hallucination"

        correct_count += int(cur_ans == item["claim_label"])
    print(correct_count/len(val_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen")
    parser.add_argument("--data_name", type=str, default="vizwiz")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--num_example", type=int, default=100)
    parser.add_argument("--eval_num_shot", type=int, default=0)
    parser.add_argument("--max_token", type=int, default=10)
    parser.add_argument("--cur_mode", type=str, default="interv")
    parser.add_argument("--activation_path", type=str, default=None)
    
    args = parser.parse_args()
    eval_reinforce(args)


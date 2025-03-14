from baukit import TraceDict, get_module
from .model import *
from .preprocess import *
import sys
import torch
import numpy as np
import json
import random
from tqdm import tqdm
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, logging
import os
from collections import Counter
from .utils import *
logging.set_verbosity_warning()



def save_mllm_encode(model, train_data, num_head):

    ###Step 1: Extract mean activation for each class label
    all_heads = model.all_heads
    #(class, head_count, resid_dim)
    print('\nExtract Mean Activations\n')
    class_activations, str_to_int, int_to_str = get_class_activations(train_data, model, all_heads)
    success_count = [0 for _ in range(class_activations.shape[1])]

    print('\nSelect Top Sparse Heads\n')
    for item in tqdm(train_data):
        #print(item)
        query_activations = get_query_activations([item], model, all_heads).squeeze(dim=0)
        int_label = str_to_int[item['label']]
        record_head_performance(class_activations, query_activations, int_label, success_count)


    ###Step 2: Pick the top-num_head based on their classification performance
    arr = np.array(success_count)
    # Get the indices of the top-k elements.
    k = num_head
    topk_indices = np.argsort(arr)[-k:][::-1]

    top_heads = []
    print("Printing Top Heads and their classification accuracy")
    for item in topk_indices.tolist():
        # TODO Map heads here
        print(item, success_count[item])
        top_heads.append(all_heads[item])

    print("\nGet Top Heads' Activations \n")
    top_class_activations, str_to_int, int_to_str = get_class_activations(train_data, model, top_heads)
    print(f'activations {top_class_activations.shape}')
    print(f'top heads {top_heads}')
    print(f'int_to_str {int_to_str}')
    return {"activations":top_class_activations, "top_heads": top_heads, "int_to_str":int_to_str}



def save_top_heads(top_heads, int_to_str, class_activations, file_path):
    """
    Saves the top heads, label mappings, and class activations to a file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = {
        "top_heads": top_heads,
        "int_to_str": int_to_str,
        "activations": class_activations.tolist()  # Convert tensor to list
    }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved top heads and class activations to {file_path}")


def load_top_heads(file_path):
    """
    Loads the top heads, label mappings, and class activations from a file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Convert list back to tensor
    class_activations = torch.tensor(data['activations']).to("cuda")
    print("Loading succesful !")
    return {
        'top_heads': data['top_heads'],
        'int_to_str': data['int_to_str'],
        'activations': class_activations
    }
    



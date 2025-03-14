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
logging.set_verbosity_warning()

def load_model(model_name, cur_dataset, lora_path=None):

    """
    A function that loads the model and a corresponding model_helper. Refer to model.py for more detail.

    Parameters:
    model_name: The name of the model you are attempting to load
    cur_dataset: The name of dataset you are attempting to load

    Returns: 
    model_helper: A helper class that contains the model as well as other functionality.
    """

    if model_name == "llava_ov_0.5b":
        ##### CUSTOM #####
        #from llava.model.builder import load_pretrained_model
        from .custom_builder import load_pretrained_model

        #pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
        pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        
        model_name = "llava_qwen"
        #device_map = "auto"
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        ##################

        llava_model_args = {
                "multimodal": True,
                # "image_aspect_ratio":"pad"
                #'load_4bit': True,
            }
        ###For finetuned models

        # overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}
        # overwrite_config = {}
        # overwrite_config["image_aspect_ratio"] = "pad"
        # llava_model_args["overwrite_config"] = overwrite_config


        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
        #tokenizer, model, image_processor, max_length = load_pretrained_model("/home/zhaobin/LLaVA-NeXT/checkpoints/Mhalu_sft", pretrained, "llava_qwen_lora", device_map=device_map, **llava_model_args)

        # ###TODO:DELETE, FOR BLINK
        # tokenizer, model, image_processor, max_length = load_pretrained_model(lora_path, pretrained, "llava_qwen_lora", device_map=device_map, **llava_model_args)

        model.eval()
        model.requires_grad_(False)
        model_helper = llavaOVDot5bHelper(model, tokenizer, image_processor, cur_dataset)
    
    elif model_name == "llava_ov_7b":
        ##### CUSTOM #####
        from llava.model.builder import load_pretrained_model
        #from .custom_builder import load_pretrained_model

        pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
        #pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        
        model_name = "llava_qwen"
        #device_map = "auto"
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        ##################

        llava_model_args = {
                "multimodal": True,
                # "image_aspect_ratio":"pad"
                #'load_4bit': True,
            }
        ###For finetuned models

        # overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064}
        # overwrite_config = {}
        # overwrite_config["image_aspect_ratio"] = "pad"
        # llava_model_args["overwrite_config"] = overwrite_config


        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
        #tokenizer, model, image_processor, max_length = load_pretrained_model("/home/zhaobin/LLaVA-NeXT/checkpoints/Mhalu_sft", pretrained, "llava_qwen_lora", device_map=device_map, **llava_model_args)

        # ###TODO:DELETE, FOR BLINK
        # tokenizer, model, image_processor, max_length = load_pretrained_model(lora_path, pretrained, "llava_qwen_lora", device_map=device_map, **llava_model_args)

        model.eval()
        model.requires_grad_(False)
        model_helper = llavaOV7bHelper(model, tokenizer, image_processor, cur_dataset)

    elif model_name == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained( "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")

        model.eval()
        model.requires_grad_(False)
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        model_helper = Qwen2Helper(model, processor, cur_dataset)

    return model_helper


def gather_last_attn_activations(inputs, model_helper):

    """
    A function that performs a forward pass and extract the activation at certain location of the layer.

    Parameters:
    inputs: input to the model. Created with model_helper
    model_helper

    Returns: 
    td: The attention activations.
    result: The output logits from forward method.
    """

    ###retain_input means the activation before passing to o_proj, which is the out projection matrix after attention. retain_out means after passing to o_proj
    ###You can decide to use something other than the o_proj by passing different config to layers. Refer to model.py
    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], retain_input=True, retain_output=True) as td:                
        #result = model_helper.generate(inputs, max_new_tokens=32)
        result = model_helper.forward(inputs)
    return td, result


def split_activations_by_head(activations, model_config):

    """
    The model concatenate the output of multi-headed attention to a single vector. This function splits this vector back to different heads.

    Parameters:
    activations: From gather_last_attn_activations
    model_config: Refer to model.py

    Returns: 
    the activation partitioned by attention heads
    """


    new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
    activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
    return activations.to("cuda")


def get_last_mean_head_activations(dataset, model_helper, N_TRIALS = 50, shot=4, no_mean=False, split="train"):

    """
    This function extracts the activation of the last input token.

    Parameters:
    dataset: a iterable item suitable for model_helper.format_func. Essentially a dataloader.
    model_helper:
    N_TRIALS: How many example to average the activation over
    shot: Number of shots per example
    no_mean: Whether you want to take the mean of the examples or save it for other preprocess

    Returns: 
    mean_activations: It has the dimension of (layer, head, Token_len, residual_dim) or (N_TRIALS, layer, head, Token_len, residual_dim). Token_len is set to 1 in this case.
    """

    activation_storage = None

    for n in range(N_TRIALS):

        #text, image_list, _, _ = model_helper.format_func(dataset, None, num_shot=shot, model_helper=model_helper, split=split)
        text, image_list, _, _ = model_helper.format_func(None, dataset[0], num_shot=shot, model_helper=model_helper, split=split)

        inputs = model_helper.insert_image(text, image_list)
        activations_td, result= gather_last_attn_activations(inputs, model_helper)


        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_helper.model_config) for layer in model_helper.model_config['attn_hook_names']]).permute(0,2,1,3)
        ###Extracting only the activation of the last input_token, as seen in the -1 indexing
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)
        if activation_storage is None:
            activation_storage = cur_activation
        else:

            activation_storage = torch.vstack((activation_storage, cur_activation))
    if no_mean:
        return activation_storage
    
    mean_activations = activation_storage.mean(dim=0)

    return mean_activations


def get_class_activations(train_dataset, model, attn_heads):

    str_to_int = {}
    int_to_str = {}
    str_to_activation = {}
    str_to_count = {}


    for item in tqdm(train_dataset):

        mean_activations = get_last_mean_head_activations([item], model, N_TRIALS = 1, shot=0)
        head_act = []
        for head in attn_heads:
            head_act.append(mean_activations[head[0], head[1], -1])
        head_act = torch.stack(head_act)

        if item['label'] in str_to_activation.keys():
            str_to_activation[item['label']] += head_act
            str_to_count[item['label']] += 1
        else:
            str_to_activation[item['label']] = head_act
            int_label = len(str_to_activation.keys()) - 1
            str_to_int[item['label']] = int_label
            int_to_str[int_label] = item['label']
            str_to_count[item['label']] = 1
    
    avg_activations = []
    for key, item in str_to_activation.items():
        avg_activations.append(torch.div(item, str_to_count[key]))
    avg_activations = torch.stack(avg_activations)

    return avg_activations, str_to_int, int_to_str


def get_query_activations(query_input, model_helper, common_heads):

    mean_activations = get_last_mean_head_activations(query_input, model_helper, N_TRIALS = 1, shot=0)
    head_act = []
    for head in common_heads:

        head_act.append(mean_activations[head[0], head[1], -1])

    head_act = torch.stack(head_act)
    
    return head_act


def record_head_performance(sample_activations, cur_activation, label, success_count):
    """
    sample_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    
    """
    #TODO change similarity here
    all_sample = []
    for i in range(sample_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activation[i, :], dim=-1)
        all_sample.append(scores.argmax(dim=0).item())
    for idx in range(len(all_sample)):
        if all_sample[idx] == label:
            success_count[idx] += 1


def retrieve_examples(sample_activations, cur_activation):

    all_sample = []
    num_heads = cur_activation.shape[0]

    for i in range(num_heads):
        scores = torch.nn.functional.cosine_similarity(
            sample_activations[:, i, :],  # (num_samples, hidden_dim)
            cur_activation[i, :],         # (hidden_dim,)
            dim=-1
        )
        all_sample.append(scores.argmax(dim=0).item())

    counter = Counter(all_sample)
    most_common = counter.most_common()

    chosen_examples = [item[0] for item in most_common]
    return chosen_examples


def mllm_encode(model, train_data, num_head):

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


def mllm_classify(inputs, model, class_embed):
    cur_activations = get_query_activations([inputs], model, class_embed['top_heads']).squeeze(dim=0)
    top_k_examples = retrieve_examples(class_embed['activations'], cur_activations)
    cur_int_label = top_k_examples[0]

    # Convert cur_int_label to the correct type
    if isinstance(cur_int_label, int) and str(cur_int_label) in class_embed['int_to_str']:
        cur_int_label = str(cur_int_label)
    elif isinstance(cur_int_label, str) and int(cur_int_label) in class_embed['int_to_str']:
        cur_int_label = int(cur_int_label)

    # Validate cur_int_label
    if cur_int_label not in class_embed['int_to_str']:
        raise KeyError(f"Invalid label index: {cur_int_label}. Valid keys are: {list(class_embed['int_to_str'].keys())}")

    return class_embed['int_to_str'][cur_int_label]



def get_activations(data, model, heads):
    """
    Precompute activations for all items in the dataset.
    Returns:
        - activations: A dictionary mapping item indices to their activations.
        - label_to_indices: A dictionary mapping labels to lists of item indices.
    """
    activations = {}
    label_to_indices = {}

    for idx, item in enumerate(tqdm(data, desc="Computing Activations")):
        mean_activations = get_last_mean_head_activations([item], model, N_TRIALS=1, shot=0)
        head_act = torch.stack([mean_activations[head[0], head[1], -1] for head in heads])
        activations[idx] = head_act

        # Map labels to indices for easy access
        if item['label'] not in label_to_indices:
            label_to_indices[item['label']] = []
        label_to_indices[item['label']].append(idx)

    return activations, label_to_indices

def get_class_activations_from_precomputed(activations, label_to_indices):
    """
    Compute average activations for each class using precomputed activations.
    """
    str_to_int = {}
    int_to_str = {}
    str_to_activation = {}
    str_to_count = {}

    for label, indices in label_to_indices.items():
        # Initialize class activations
        str_to_activation[label] = torch.zeros_like(activations[indices[0]])
        str_to_count[label] = 0

        # Sum activations for the class
        for idx in indices:
            str_to_activation[label] += activations[idx]
            str_to_count[label] += 1

        # Map labels to integers
        int_label = len(str_to_int)
        str_to_int[label] = int_label
        int_to_str[int_label] = label

    # Compute average activations
    avg_activations = torch.stack([torch.div(str_to_activation[label], str_to_count[label]) for label in str_to_activation])
    return avg_activations, str_to_int, int_to_str

def optimized_mllm_encode(model, train_data, num_head):
    all_heads = model.all_heads

    # Step 1: Precompute activations for all items
    activations, label_to_indices = get_activations(train_data, model, all_heads)

    # Step 2: Compute class activations using precomputed data
    class_activations, str_to_int, int_to_str = get_class_activations_from_precomputed(activations, label_to_indices)

    # Step 3: Evaluate head performance
    success_count = [0 for _ in range(len(all_heads))]
    for idx, item in enumerate(tqdm(train_data, desc="Evaluating Heads")):
        query_activations = activations[idx]
        int_label = str_to_int[item['label']]
        record_head_performance(class_activations, query_activations, int_label, success_count)

    # Step 4: Select top heads based on performance
    topk_indices = np.argsort(success_count)[-num_head:][::-1]
    top_heads = [all_heads[idx] for idx in topk_indices]

    print("Top Heads and their classification accuracy:")
    for idx in topk_indices:
        print(f"Head {idx}: {success_count[idx]}")

    # Step 5: Compute activations for top heads
    top_class_activations, _, _ = get_class_activations_from_precomputed(
        {idx: activations[idx] for idx in range(len(train_data))}, label_to_indices
    )

    print(f"Activations shape: {top_class_activations.shape}")
    print(f"Top Heads: {top_heads}")
    print(f"Label Mapping: {int_to_str}")

    return {
        "activations": top_class_activations,
        "top_heads": top_heads,
        "int_to_str": int_to_str,
    }


def save_top_heads(top_heads, int_to_str, file_path):
    """
    Saves the selected top attention heads and label mappings to a file.
    Creates the directory structure if it does not exist.

    Parameters:
    - file_path (str): Path to the file where data should be saved.
    - top_heads (list): List of selected top heads as (layer, head) tuples.
    - int_to_str (dict): Mapping from integer labels to class names.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Prepare data to save
    data = {
        "top_heads": top_heads,
        "int_to_str": int_to_str
    }

    # Save data to file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)  # Use indent for pretty-printing

    print(f"Top heads saved to {file_path}")


def load_top_heads(file_path):
    """
    Load the top heads and their corresponding label mappings from a JSON file,
    and return them in a class_embed dictionary format.

    Parameters:
    file_path (str): The path to the file that stores the top heads and int_to_str.

    Returns:
    dict: A dictionary containing 'top_heads' and 'int_to_str'.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Load data from file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Prepare the return dictionary
    class_embed = {
        'top_heads': data['top_heads'],
        'int_to_str': data['int_to_str']
    }

    print(f"Top heads loaded from {file_path}")
    return class_embed
    


def precompute_all_activations(data, model, top_heads):
    """
    Precompute activations for all items in the dataset.

    Parameters:
    - data: The dataset (list of items).
    - model: The pre-trained model.
    - top_heads: List of top heads to use for computing activations.

    Returns:
    - query_activations: A dictionary mapping item indices to their query activations.
    - class_activations: The average activations for each class.
    - int_to_str: Mapping from integer labels to class names.
    """
    query_activations = {}
    str_to_activation = {}
    str_to_count = {}
    str_to_int = {}
    int_to_str = {}

    for idx, item in enumerate(tqdm(data, desc="Precomputing Activations")):
        # Compute query activations
        mean_activations = get_last_mean_head_activations([item], model, N_TRIALS=1, shot=0)
        head_act = torch.stack([mean_activations[head[0], head[1], -1] for head in top_heads])
        query_activations[idx] = head_act

        # Compute class activations
        if item['label'] in str_to_activation.keys():
            str_to_activation[item['label']] += head_act
            str_to_count[item['label']] += 1
        else:
            str_to_activation[item['label']] = head_act
            int_label = len(str_to_activation.keys()) - 1
            str_to_int[item['label']] = int_label
            int_to_str[int_label] = item['label']
            str_to_count[item['label']] = 1

    # Compute average class activations
    class_activations = torch.stack([torch.div(str_to_activation[key], str_to_count[key]) for key in str_to_activation])

    return query_activations, class_activations, int_to_str


def inference_mllm_classify(query_activations, class_activations, int_to_str, idx):
    """
    Classify an input using precomputed activations.

    Parameters:
    - query_activations: Precomputed activations for all queries.
    - class_activations: Precomputed activations for all classes.
    - int_to_str: Mapping from integer labels to class names.
    - idx: Index of the query in the dataset.

    Returns:
    - Predicted class label.
    """
    cur_activations = query_activations[idx]
    top_k_examples = retrieve_examples(class_activations, cur_activations)
    cur_int_label = top_k_examples[0]

    # Convert cur_int_label to the correct type
    if isinstance(cur_int_label, int) and str(cur_int_label) in int_to_str:
        cur_int_label = str(cur_int_label)
    elif isinstance(cur_int_label, str) and int(cur_int_label) in int_to_str:
        cur_int_label = int(cur_int_label)

    # Validate cur_int_label
    if cur_int_label not in int_to_str:
        raise KeyError(f"Invalid label index: {cur_int_label}. Valid keys are: {list(int_to_str.keys())}")

    return int_to_str[cur_int_label]

###########################################
#############PER CLASS PART################
###########################################

def mllm_encode_per_class(model, train_data, num_heads_per_class=20):
    """
    Selects top K heads for each class based on class-specific discriminative performance.
    Returns a dictionary with class embeddings containing their top heads and activations.
    """
    all_heads = model.all_heads
    num_heads = len(all_heads)
    
    # Step 1: Compute initial class activations with all heads
    print("Computing class activations...")
    class_activations_all, str_to_int, int_to_str = get_class_activations(train_data, model, all_heads)
    num_classes = len(int_to_str)
    
    # Step 2: Initialize class-specific success counts
    class_success_counts = {c: [0]*num_heads for c in range(num_classes)}
    
    # Step 3: Evaluate head performance per class
    print("Evaluating heads per class...")
    for item in tqdm(train_data):
        int_label = str_to_int[item['label']]
        query_activations = get_query_activations([item], model, all_heads).squeeze(0)
        
        # For each head, check if it predicts the correct class
        for head_idx in range(num_heads):
            head_activations = class_activations_all[:, head_idx, :]  # (num_classes, hidden_dim)
            scores = torch.nn.functional.cosine_similarity(
                head_activations, 
                query_activations[head_idx].unsqueeze(0),
                dim=-1
            )
            predicted_class = scores.argmax().item()
            if predicted_class == int_label:
                class_success_counts[int_label][head_idx] += 1
    
    # Step 4: Select top heads per class
    class_embeddings = {}
    for class_idx in range(num_classes):
        counts = np.array(class_success_counts[class_idx])
        topk_indices = np.argsort(counts)[-num_heads_per_class:][::-1].tolist()
        
        # Store indices rather than tuples for efficient slicing
        class_label = int_to_str[class_idx]
        class_embeddings[class_label] = {
            'top_head_indices': topk_indices,
            'top_heads': [all_heads[i] for i in topk_indices],
            'int_to_str': int_to_str
        }
    
    # Step 5: Compute class-specific activations for selected heads
    print("Computing final class activations...")
    for class_label in tqdm(class_embeddings):
        heads = class_embeddings[class_label]['top_heads']
        class_activations, _, _ = get_class_activations(train_data, model, heads)
        class_embeddings[class_label]['activations'] = class_activations
        
    return class_embeddings

def mllm_classify_per_class(inputs, model, class_embeddings):
    """
    Classifies input using a voting mechanism where each class's head set votes independently.
    Returns the class with the most votes.
    """
    # Get all heads' activations once
    all_heads = model.all_heads
    query_activations_all = get_query_activations([inputs], model, all_heads).squeeze(0)
    
    votes = defaultdict(int)
    
    for class_label, embedding in class_embeddings.items():
        # Get relevant activations for this class's heads
        head_indices = embedding['top_head_indices']
        activations = query_activations_all[head_indices]
        class_avg_activations = embedding['activations']
        
        # Count votes from this class's heads
        for head_idx in range(len(head_indices)):
            head_act = activations[head_idx]
            scores = torch.nn.functional.cosine_similarity(
                class_avg_activations[:, head_idx, :],
                head_act.unsqueeze(0),
                dim=-1
            )
            predicted_class = embedding['int_to_str'][scores.argmax().item()]
            if predicted_class == class_label:
                votes[class_label] += 1
                
    return max(votes.items(), key=lambda x: x[1])[0] if votes else None


def save_class_embeddings(class_embeddings, file_path):
    """Serializes class embeddings to JSON format"""
    serializable = {}
    for cls, data in class_embeddings.items():
        serializable[cls] = {
            'top_head_indices': data['top_head_indices'],
            'top_heads': [list(h) for h in data['top_heads']],  # Convert tuples to lists
            'int_to_str': data['int_to_str'],
            # Note: Activations tensor is not serialized - use precompute_and_save_activations()
        }
    with open(file_path, 'w') as f:
        json.dump(serializable, f)

def load_class_embeddings(file_path):
    """
    Load class-specific embeddings from JSON file.
    
    Parameters:
    file_path (str): Path to saved class embeddings
    
    Returns:
    dict: Nested dictionary containing per-class configurations
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Class embeddings file {file_path} not found")

    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    # Validate and convert data format
    class_embeddings = {}
    int_to_str = None
    
    for class_name, class_data in raw_data.items():
        # Convert list indices to tuples for head locations
        class_embeddings[class_name] = {
            'top_head_indices': class_data['top_head_indices'],
            'top_heads': [tuple(head) for head in class_data['top_heads']],
            'int_to_str': {int(k): v for k, v in class_data['int_to_str'].items()}
        }
        
        # Verify consistent label mapping across classes
        if int_to_str is None:
            int_to_str = class_embeddings[class_name]['int_to_str']
        elif class_embeddings[class_name]['int_to_str'] != int_to_str:
            raise ValueError("Inconsistent int_to_str mappings between classes")

    # Add shared label mapping to root
    class_embeddings['int_to_str'] = int_to_str
    
    print(f"Loaded class embeddings for {len(raw_data)} classes from {file_path}")
    return class_embeddings


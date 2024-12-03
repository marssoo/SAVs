
from baukit import TraceDict, get_module
from models import *
from preprocess import *
import sys
import torch
import numpy as np
import json
import random
from tqdm import tqdm
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, logging
import sys
from torchvision.ops.boxes import box_area
# from pycocoevalcap.eval import COCOEvalCap
# from pycocotools.coco import COCO
from collections import Counter
logging.set_verbosity_warning()
import warnings
import requests
torch.autograd.set_detect_anomaly(True)
sys.path.append('../eval_mm')
from vqa import VQA
from vqa_eval import VQAEval


def load_model(model_name, cur_dataset, lora_path=None):

    """
    A function that loads the model and a corresponding model_helper. Refer to model.py for more detail.

    Parameters:
    model_name: The name of the model you are attempting to load
    cur_dataset: The name of dataset you are attempting to load

    Returns: 
    model_helper: A helper class that contains the model as well as other functionality.
    """

    if model_name == "llava_ov":
        from llava.model.builder import load_pretrained_model
        
        pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"

        
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        llava_model_args = {
                "multimodal": True,
            }
        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
        model.eval()
        model.requires_grad_(False)
        model_helper = llavaOVHelper(model, tokenizer, image_processor, cur_dataset)

    elif model_name == "qwen2":
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
        ###TODO: DELETE ME. GETTING THE FIRST TOKEN
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)
        if activation_storage is None:
            activation_storage = cur_activation
        else:

            activation_storage = torch.vstack((activation_storage, cur_activation))
    if no_mean:
        return activation_storage
    
    mean_activations = activation_storage.mean(dim=0)

    return mean_activations


def get_last_mean_layer_activations(dataset, model_helper, N_TRIALS = 50, shot=4, no_mean=False, split="train"):
    activation_storage = None

    for n in range(N_TRIALS):

        #text, image_list, _, _ = model_helper.format_func(dataset, None, num_shot=shot, model_helper=model_helper, split=split)
        text, image_list, _, _ = model_helper.format_func(None, dataset[0], num_shot=shot, model_helper=model_helper, split=split)

        inputs = model_helper.insert_image(text, image_list)
        activations_td, result= gather_last_attn_activations(inputs, model_helper)

        ###(layer, seq_len, resid_dim)
        stack_initial = torch.vstack([activations_td[layer].output.to("cuda") for layer in model_helper.model_config['attn_hook_names']])
        ###(1, layer, seq_len, resid_dim). Pretend layer to be head, and layer=1
        stack_initial = stack_initial.unsqueeze(dim=0)
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)


        if activation_storage is None:
            activation_storage = cur_activation
        else:

            activation_storage = torch.vstack((activation_storage, cur_activation))
    if no_mean:
        return activation_storage
    
    mean_activations = activation_storage.mean(dim=0)

    return mean_activations


def get_sample_activations(train_dataset, model_helper, common_heads, is_knn=False, is_layer=False):

    all_act = []
    for item in train_dataset:

        if is_layer:
            mean_activations = get_last_mean_layer_activations([item], model_helper, N_TRIALS = 1, shot=0)
        else:
            mean_activations = get_last_mean_head_activations([item], model_helper, N_TRIALS = 1, shot=0)

        head_act = []
        for head in common_heads:

            head_act.append(mean_activations[head[0], head[1], -1])

        head_act = torch.stack(head_act)
        all_act.append(head_act)

    if is_knn:
        return torch.stack(all_act)
    else:
        return torch.stack(all_act).mean(dim=0, keepdim=True)


def get_blink_sample_activations(train_dataset, model_helper, common_heads, is_layer=False, use_knn=False):

    A_act = []
    B_act = []
    C_act = []
    D_act = []

    count = 0
    for item in tqdm(train_dataset):

        if count == 40:
            break
        count += 1

        if is_layer:
            mean_activations = get_last_mean_layer_activations([item], model_helper, N_TRIALS = 1, shot=0)
        else:
            mean_activations = get_last_mean_head_activations([item], model_helper, N_TRIALS = 1, shot=0)
        head_act = []
        for head in common_heads:

            head_act.append(mean_activations[head[0], head[1], -1])

        head_act = torch.stack(head_act)

        if item["answer"] == "(A)":
            A_act.append(head_act)
        elif item["answer"] == "(B)":
            B_act.append(head_act)
        elif item["answer"] == "(C)":
            C_act.append(head_act)
        else:
            D_act.append(head_act)

    combined_act = []
    for choice_act in [A_act, B_act, C_act, D_act]:
        if choice_act != []:
            if use_knn:
                combined_act.append(torch.stack(choice_act))
            else:
                combined_act.append(torch.stack(choice_act).mean(dim=0, keepdim=True))

    return torch.vstack(combined_act)


def natural_scorer(all_correct_count):
    q_score = 0
    i_score = 0
    g_score = 0

    if all_correct_count[0] == 1 and all_correct_count[2] == 1:
        q_score += 0.5
    if all_correct_count[3] == 1 and all_correct_count[1] == 1:
        q_score += 0.5
    
    if all_correct_count[0] == 1 and all_correct_count[1] == 1:
        i_score += 0.5
    if all_correct_count[3] == 1 and all_correct_count[2] == 1:
        i_score += 0.5

    if q_score == 1 and i_score == 1:
        g_score += 1
    
    return q_score, i_score, g_score


def natural_splitter(sample, idx):

    img_0_split = [(sample["Image_0"][idx], sample["Question_0"][idx], sample["Image_0_Question_0"][idx]), (sample["Image_0"][idx], sample["Question_1"][idx], sample["Image_0_Question_1"][idx])]
    img_1_split = [(sample["Image_1"][idx], sample["Question_0"][idx], sample["Image_1_Question_0"][idx]), (sample["Image_1"][idx], sample["Question_1"][idx], sample["Image_1_Question_1"][idx])]
    return img_0_split + img_1_split


def get_natural_sample_activations(train_dataset, model_helper, common_heads, q_type):

    yes_act = []
    no_act = []


    for idx in tqdm(range(len(train_dataset["Index"]))):

        splitted_natural = natural_splitter(train_dataset, idx)

        for cur_split in splitted_natural:

            if train_dataset["Question_Type"][idx] != q_type:
                continue

            mean_activations = get_last_mean_head_activations([cur_split], model_helper, N_TRIALS = 1, shot=0)
            head_act = []
            for head in common_heads:

                head_act.append(mean_activations[head[0], head[1], -1])

            head_act = torch.stack(head_act)
            if cur_split[2] == "Yes" or cur_split[2] == "A":
                yes_act.append(head_act)
            else:
                no_act.append(head_act)

    return torch.vstack((torch.stack(yes_act).mean(dim=0, keepdim=True),torch.stack(no_act).mean(dim=0, keepdim=True)))


def get_natural_balance_sample_activations(train_dataset, model_helper, common_heads, is_layer=False, is_knn=False):
    yes_act = []
    no_act = []

    for item in tqdm(train_dataset):

        if is_layer:
            mean_activations = get_last_mean_layer_activations([item], model_helper, N_TRIALS = 1, shot=0)
        else:
            mean_activations = get_last_mean_head_activations([item], model_helper, N_TRIALS = 1, shot=0)

        head_act = []
        for head in common_heads:

            head_act.append(mean_activations[head[0], head[1], -1])

        head_act = torch.stack(head_act)

        if item[2] == "Yes" or item[2] == "A":
            yes_act.append(head_act)
        else:
            no_act.append(head_act)

    if is_knn:
        return torch.vstack((torch.stack(yes_act),torch.stack(no_act)))
    else:
        return torch.vstack((torch.stack(yes_act).mean(dim=0, keepdim=True),torch.stack(no_act).mean(dim=0, keepdim=True)))


def get_wino_sample_activations(train_dataset, model_helper, common_heads, is_layer=False):

    yes_act = []
    no_act = []

    for item in tqdm(train_dataset):

        item = json.loads(item)

        if is_layer:
            mean_activations = get_last_mean_layer_activations([item], model_helper, N_TRIALS = 1, shot=0)
        else:
            mean_activations = get_last_mean_head_activations([item], model_helper, N_TRIALS = 1, shot=0)

        head_act = []
        for head in common_heads:

            head_act.append(mean_activations[head[0], head[1], -1])

        head_act = torch.stack(head_act)

        if item["answer"] == "Yes":
            yes_act.append(head_act)
        else:
            no_act.append(head_act)

    combined_act = []
    for choice_act in [yes_act, no_act]:
        if choice_act != []:
            combined_act.append(torch.stack(choice_act).mean(dim=0, keepdim=True))
    
    return torch.vstack(combined_act)


def get_mmmu_sample_activations(collected_dev_examples, model_helper, common_heads):
    A_act = []
    B_act = []
    C_act = []
    D_act = []

    for item in tqdm(collected_dev_examples):
        mean_activations = get_last_mean_head_activations([item], model_helper, N_TRIALS = 1, shot=0)
        head_act = []
        for head in common_heads:

            head_act.append(mean_activations[head[0], head[1], -1])

        head_act = torch.stack(head_act)

        if item["answer"] == "A":
            A_act.append(head_act)
        elif item["answer"] == "B":
            B_act.append(head_act)
        elif item["answer"] == "C":
            C_act.append(head_act)
        elif item["answer"] == "D":
            D_act.append(head_act)


    combined_act = []
    for choice_act in [A_act, B_act, C_act, D_act]:
        if choice_act != []:
            combined_act.append(torch.stack(choice_act).mean(dim=0, keepdim=True))
    
    return torch.vstack(combined_act)


def get_eurosat_sample_activations(train_dataset, model_helper, common_heads, is_layer=False, use_knn=False):
    combined_act = []
    idx_to_label = {}
    label_to_idx = {}

    count = 0
    for key, q_list in tqdm(train_dataset.items()):
        idx_to_label[count] = key
        label_to_idx[key] = count

        count += 1

        current_act = []
        for item in q_list:
            
            if is_layer:
                mean_activations = get_last_mean_layer_activations([item], model_helper, N_TRIALS = 1, shot=0)
            else:
                mean_activations = get_last_mean_head_activations([item], model_helper, N_TRIALS = 1, shot=0)
            head_act = []
            for head in common_heads:
                head_act.append(mean_activations[head[0], head[1], -1])
            head_act = torch.stack(head_act)
            current_act.append(head_act)

        if use_knn:
            combined_act.append(torch.stack(current_act))
        else:
            combined_act.append(torch.stack(current_act).mean(dim=0, keepdim=True))
    
    return torch.vstack(combined_act), idx_to_label, label_to_idx


###exactly the same as eurosat
def get_airplane_sample_activations(train_dataset, model_helper, common_heads, is_layer=False):
    combined_act = []
    idx_to_label = {}
    label_to_idx = {}

    count = 0
    for key, q_list in train_dataset.items():
        idx_to_label[count] = key
        label_to_idx[key] = count

        count += 1

        current_act = []
        for item in q_list:
            
            if is_layer:
                mean_activations = get_last_mean_layer_activations([item], model_helper, N_TRIALS = 1, shot=0)
            else:
                mean_activations = get_last_mean_head_activations([item], model_helper, N_TRIALS = 1, shot=0)
            head_act = []
            for head in common_heads:
                head_act.append(mean_activations[head[0], head[1], -1])
            head_act = torch.stack(head_act)
            current_act.append(head_act)

        combined_act.append(torch.stack(current_act).mean(dim=0, keepdim=True))
    
    return torch.vstack(combined_act), idx_to_label, label_to_idx


def get_language_sample_activations(train_dataset, model_helper, common_heads, is_layer=False, is_knn=False):
    neg_act = []
    pos_act = []

    neu_act = []
    ent_act = []
    con_act = []

    for item in tqdm(train_dataset):

        mean_activations = get_last_mean_head_activations([item], model_helper, N_TRIALS = 1, shot=0)

        head_act = []
        for head in common_heads:

            head_act.append(mean_activations[head[0], head[1], -1])

        head_act = torch.stack(head_act)

        # if item['label'] == "negative":
        #     neg_act.append(head_act)
        # else:
        #     pos_act.append(head_act)


        if item['label'] == "neutral":
            neu_act.append(head_act)
        elif item['label'] == "entailment":
            ent_act.append(head_act)
        else:
            con_act.append(head_act)

    return torch.vstack((torch.stack(neu_act).mean(dim=0, keepdim=True), torch.stack(ent_act).mean(dim=0, keepdim=True), torch.stack(con_act).mean(dim=0, keepdim=True)))
    #return torch.vstack((torch.stack(pos_act).mean(dim=0, keepdim=True),torch.stack(neg_act).mean(dim=0, keepdim=True)))


def record_head_performance(sample_activations, cur_activation, label, success_count, is_knn=False):
    """
    sample_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    
    """

    all_sample = []
    for i in range(sample_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activation[i, :], dim=-1)

        if is_knn:

            topk = torch.topk(scores, k=5, dim=0)[1] // 20
            all_sample.append(torch.max(topk).item())

        else:
            all_sample.append(scores.argmax(dim=0).item())
    for idx in range(len(all_sample)):
        if all_sample[idx] == label:
            success_count[idx] += 1


def retrieve_examples(sample_activations, cur_activation, is_knn=False, is_online=False):
    """
    sample_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    
    """

    all_sample = []

    for i in range(sample_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activation[i, :], dim=-1)
        if is_knn:
            topk = torch.topk(scores, k=5, dim=0)[1]

            topk = topk // 20
            all_sample.append(torch.max(topk).item())
        else:
            all_sample.append(scores.argmax(dim=0).item())
    if is_online:
        return all_sample
    
    counter = Counter(all_sample)
    most_common = counter.most_common()

    chosen_examples = []
    for item in most_common:
        chosen_examples.append(item[0])
    return chosen_examples


def retrieve_examples_with_count(sample_activations, cur_activation):
    """
    sample_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    
    """

    all_sample = []
    positive_count = 0
    yes_sim = []
    for i in range(sample_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activation[i, :], dim=-1)
        argmax_label = scores.argmax(dim=0).item()
        all_sample.append(argmax_label)

        yes_sim.append(torch.sigmoid(scores)[0].item())
        
    yes_sim = torch.tensor(yes_sim).mean().item()

    counter = Counter(all_sample)
    most_common = counter.most_common()

    chosen_examples = []
    for item in most_common:
        ###Return both the class and number of head classified this
        chosen_examples.append(item[0])
        if item[0] == 0:
            positive_count = item[1]
    return chosen_examples, positive_count, yes_sim



def retrieve_examples_k_heads(sample_activations, cur_activation, num_head):
    """
    sample_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    
    """

    all_pred = []
    for cur_k in num_head:
        all_sample = []
        for i in range(cur_k):
            scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activation[i, :], dim=-1)

            all_sample.append(scores.argmax(dim=0).item())

        counter = Counter(all_sample)
        most_common = counter.most_common(1)
        all_pred.append(most_common[0][0])

    return all_pred

def get_query_activations(train_dataset, model_helper, common_heads, is_layer=False):
    all_head_act = []
    for item in train_dataset:

        if is_layer:
            mean_activations = get_last_mean_layer_activations([item], model_helper, N_TRIALS = 1, shot=0)
        else:
            mean_activations = get_last_mean_head_activations([item], model_helper, N_TRIALS = 1, shot=0)

        head_act = []
        for head in common_heads:

            head_act.append(mean_activations[head[0], head[1], -1])

        head_act = torch.stack(head_act)
        all_head_act.append(head_act)
    
    return torch.stack(all_head_act)


def naive_bayes_classifier(sample_activations, cur_activations, log_prob_table, class_log_prob):

    """
    
    log_prob_table: (feature, class) THe probably of the choice being 1 given a class
    """

    ###len = 40
    all_sample = []
    for i in range(sample_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activations[i, :], dim=-1)

        all_sample.append(scores.argmax(dim=0).item())


    ###(index, conditional prob)
    stored_max = (-1, -999)
    for class_idx in range(sample_activations.shape[0]):
        log_prob_sum = 0
        for head_idx in range(len(all_sample)):
            if all_sample[head_idx] == 1:
                log_prob_sum += log_prob_table[head_idx, class_idx]
            else:
                log_prob_sum += 1 - log_prob_table[head_idx, class_idx]

        log_prob_sum += class_log_prob[class_idx]
        if log_prob_sum > stored_max[1]:
            stored_max = (class_idx, log_prob_sum)

    return stored_max


def get_conditional_table(sample_activations, cur_activation, label, prob_table):
    """
    sample_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    prob_table: (feature, class) THe probably of a head activated given a class
    
    """

    all_sample = []
    for i in range(sample_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activation[i, :], dim=-1)

        all_sample.append(scores.argmax(dim=0).item())

    for idx in range(len(all_sample)):
        prob_table[idx, label] += all_sample[idx]


def get_pred_threshold(sample_activations, cur_activation, label, threshold):
    """
    sample_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    
    """

    all_sample = []


    for i in range(sample_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activation[i, :], dim=-1)

        all_sample.append(scores.argmax(dim=0).item())

    counter = Counter(all_sample)
    most_common = counter.most_common()

    for item in most_common:
        if item[0] == label:
            threshold[label] += item[1]


def retrieve_examples_threshold(sample_activations, cur_activation, threshold):
    """
    sample_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    
    """

    all_sample = []

    difference =[-999, -999]

    for i in range(sample_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(sample_activations[:, i, :], cur_activation[i, :], dim=-1)

        all_sample.append(scores.argmax(dim=0).item())

    counter = Counter(all_sample)
    most_common = counter.most_common()

    for item in most_common:
        difference[item[0]] = item[1] - threshold[item[0]]

    return [np.argmax(difference)]


def softmax(logits):
    """Convert logits to probabilities using the softmax function."""
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def sample_from_logits(logits, num_samples=1):
    """
    Sample indices based on logits using softmax probabilities.
    
    Args:
        logits: Array-like of logit values
        num_samples: Number of samples to draw (default: 1)
        
    Returns:
        If num_samples=1, returns a single sampled index
        If num_samples>1, returns an array of sampled indices
    """
    probs = softmax(logits)
    indices = np.arange(len(logits))
    samples = np.random.choice(indices, size=num_samples, p=probs)
    
    return samples[0] if num_samples == 1 else samples



def online_updates(online_data, combined_activations, new_heads, model_helper):
    eps = np.sqrt(np.log10(20)/100)
    head_weight = [1 for _ in range(20)]

    for item in tqdm(online_data):

        cur_activations = get_query_activations([item], model_helper, new_heads).squeeze(dim=0)
        all_sample = retrieve_examples(combined_activations, cur_activations, is_online=True)

        if item["claim_label"] == "hallucination":
            gt_label = 0
        else:
            gt_label = 1
        
        for _ in range(20):
            if all_sample[_] != gt_label:
                head_weight[_] = (1-eps)*head_weight[_]
    
    return head_weight
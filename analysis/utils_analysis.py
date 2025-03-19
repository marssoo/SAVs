import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

np.random.seed(0)

#### Loading or skipping test
def load_or_skip(path, validation, validation_threshold=1600):
    number_of_chunks = 0
    for file in os.listdir(path):
        if 'test_activations' in file:
            number_of_chunks += 1
    test = {}
    for i in range(number_of_chunks):
        # load chunk
        with open(path + f'/test_activations_{i}.pkl', 'rb') as f:
            test_chunk = pickle.load(f)
            test.update(test_chunk)

    #if we're doing validation and the dataset is big enough :
    if validation:
        if len(test) < validation_threshold:
            return None, None, True #if too small we skip
        num_to_keep = int(0.25 * len(test)) #else we go on with the first 25%

        if 'natural_ret' in path: #if it's natural ret we need to keep groups together
            selected_keys = range(num_to_keep // 4 * 4)
        else:
            selected_keys = range(num_to_keep)
    else:
        num_to_keep = int(0.25 * len(test))
        if 'natural_ret' in path: 
            selected_keys = range(num_to_keep // 4 * 4, len(test))
        else:
            selected_keys = range(num_to_keep, len(test))

    new_test = {key: test[key] for key in selected_keys}
    test = new_test
    del new_test
    
    #load ground_truth once and for all
    with open(path + '/test_classes.pkl', 'rb') as f:
        test_labels_to_indices = pickle.load(f)
    
    return test, test_labels_to_indices, False
            
#### scores ######################################

def record_head_performance_base(class_activations, cur_activation, label, success_count, hp1_useless, hp2_useless):
    """
    class_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    success_count is dynamically updated
    """
    all_sample = []

    for i in range(class_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(class_activations[:, i, :], cur_activation[i, :], dim=-1)
        all_sample.append(scores.argmax(dim=0).item())
    for idx in range(len(all_sample)):
        if all_sample[idx] == label:
            success_count[idx] += 1
    return success_count

def record_head_performance_polar(class_activations, cur_activation, label, success_count, hp1_useless, hp2_useless):
    """
    class_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    success_count is dynamically updated
    """
    all_sample = []

    all_sample = []
    num_classes = class_activations.shape[0]
    for i in range(class_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(class_activations[:, i, :], cur_activation[i, :], dim=-1)
        scores_hat = torch.zeros_like(scores)
        ### Substracting other classes' scores
        for j in range(num_classes):
            other_classes_sum = scores.sum(dim=0) - scores[j]
            scores_hat[j] = scores[j] - (other_classes_sum / (num_classes - 1))

        all_sample.append(scores_hat.argmax(dim=0).item())
    #print(len(all_sample))
    for idx in range(len(all_sample)):
        if all_sample[idx] == label:
            success_count[idx] += 1

    return success_count


def score_arctanh(x, y, alpha=0.45, beta=0.3):
        """score with arctanh"""
        score = ((1+torch.atanh(x))*alpha - torch.atanh(y)*beta)
        return score

def record_head_performance_artanh(class_activations, cur_activation, label, success_count, alpha, beta):
    """
    class_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    success_count is dynamically updated
    """
    all_sample = []
    num_classes = class_activations.shape[0]
    for i in range(class_activations.shape[1]):
        scores = torch.nn.functional.cosine_similarity(class_activations[:, i, :], cur_activation[i, :], dim=-1)
        scores_hat = torch.zeros_like(scores)
        ### Substracting other classes' scores
        #print(scores.shape)
        for j in range(num_classes):
            other_classes_sum = scores.sum(dim=0) - scores[j]
            x = scores[j]
            y = (other_classes_sum / (num_classes - 1))
            scores_hat[j] = score_arctanh(x, y, alpha, beta)

        all_sample.append(scores_hat.argmax(dim=0).item())
    #print(len(all_sample))
    for idx in range(len(all_sample)):
        if all_sample[idx] == label:
            success_count[idx] += 1

    return success_count


##################################################

def get_success_counts(class_activations, all_activations, str_to_int, indices_to_labels, score, hp1, hp2):
    success_count = [0 for _ in range(class_activations.shape[1])]
    #match score function:
    if score == 'base':
        record_head_performance = record_head_performance_base
    elif score == 'polar':
        record_head_performance = record_head_performance_polar
    elif score == 'artanh':
        record_head_performance = record_head_performance_artanh

    #go through training data
    for index, activation in all_activations.items():
        int_label = str_to_int[indices_to_labels[index]]
        success_count = record_head_performance(class_activations, activation, int_label, success_count, hp1, hp2)

    return np.array(success_count)

def get_top_heads(all_heads, topk_indices):
    #assuming topk_indices sorted
    k = len(topk_indices)
    if len(all_heads.shape) == 3:
        top_heads = torch.zeros((all_heads.shape[0], k, all_heads.shape[2]))
        for i, k in enumerate(topk_indices):
            top_heads[:, i, :] = all_heads[:, k, :]
        return top_heads

    elif len(all_heads.shape) != 2:
        raise ValueError("Unrecognized shape for activations")
        
    top_heads = torch.zeros((k, all_heads.shape[1]))
    for i, k in enumerate(topk_indices):
        top_heads[i, :] = all_heads[k, :]
    return top_heads

def retrieve_examples(sample_activations, cur_activation):
    """sample_activations = class_activations limited to the top heads"""
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

# Evaluation function
def evaluate_natural_ret(test, benchmark, activations, topk_indices, top_class_activations, int_to_str, test_labels_to_indices):
    test_size = len(test)
    results = {}
    for index, activations in test.items():
        top_heads = get_top_heads(activations, topk_indices)  # only the heads we want
        preds = retrieve_examples(top_class_activations, top_heads)
        pred = preds[0]
        str_class = int_to_str[pred]  # convert integer prediction to string label
        # Set 1 if the current sample index is in the set of indices for the predicted label, else 0
        results[index] = int(index in test_labels_to_indices[str_class])

    results = np.array(list(results.values()))
    # Reshape results into groups of 4 for further aggregated metrics
    group_results = results.reshape(-1, 4)
    total_groups = group_results.shape[0]
    # Raw accuracy: average correctness across all samples
    raw_acc = results.mean()
    raw_std = results.std()
    # Question accuracy:
    # For each group, the first question is correct if both sample 0 and 1 are correct,
    # and the second question is correct if both sample 2 and 3 are correct.
    q_first = group_results[:, 0] * group_results[:, 1]
    q_second = group_results[:, 2] * group_results[:, 3]
    q_correct = np.sum(q_first) + np.sum(q_second)
    q_acc = q_correct / (total_groups * 2)  # Two questions per group
    # Create a combined array of question outcomes per group (0 or 1 per question)
    q_outcomes = np.concatenate((q_first, q_second))
    q_std = q_outcomes.std()
    # Image accuracy:
    # First image is correct if samples 0 and 2 are correct;
    # Second image is correct if samples 1 and 3 are correct.
    i_first = group_results[:, 0] * group_results[:, 2]
    i_second = group_results[:, 1] * group_results[:, 3]
    i_correct = np.sum(i_first) + np.sum(i_second)
    i_acc = i_correct / (total_groups * 2)  # Two images per group
    # Create a combined array of image outcomes per group (0 or 1 per image)
    i_outcomes = np.concatenate((i_first, i_second))
    i_std = i_outcomes.std()
    # Group accuracy: a group is correct if all four samples are correct.
    g_outcomes = np.all(group_results == 1, axis=1).astype(float)
    g_acc = g_outcomes.mean()
    g_std = g_outcomes.std()
    # Print the metrics with standard deviations
    #print(f"Raw Accuracy: {raw_acc:.4f}") #± {raw_std:.4f}")
    #print(f"Question Accuracy: {q_acc:.4f}") #± {q_std:.4f}")
    #print(f"Image Accuracy: {i_acc:.4f}") #± {i_std:.4f}")
    #print(f"Group Accuracy: {g_acc:.4f}") #± {g_std:.4f}")
    return raw_acc, raw_std, q_acc, q_std, i_acc, i_std, g_acc, g_std

def evaluate(test, benchmark, activations, topk_indices, top_class_activations, int_to_str, test_labels_to_indices):
    test_size = len(test)
    results = {} # will hold results, we just need one dim

    for index, activations in test.items():
        top_heads = get_top_heads(activations, topk_indices)        #only the heads we want
        preds = retrieve_examples(top_class_activations, top_heads) 
        pred = preds[0]
        str_class = int_to_str[pred]                              #string prediction
        #print(index)
        results[index] = int(index in test_labels_to_indices[str_class])  # 0 if the index isn't in the predicted class' indexes
    
    results = np.array(list(results.values()))
    accuracy = results.mean()
    std = results.std()
    return accuracy, std
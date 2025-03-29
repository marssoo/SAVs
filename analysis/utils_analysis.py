import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from groupyr import LogisticSGL  # pip install groupyr, needs to downgrade sklearn to 1.4 (--force-reinstall)

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

    for key in list(test.keys()):
        if key not in selected_keys:
            del test[key]
    #new_test = {key: test[key] for key in selected_keys}
    #test = new_test
    #del new_test
    
    #load ground_truth once and for all
    with open(path + '/test_classes.pkl', 'rb') as f:
        test_labels_to_indices = pickle.load(f)
    
    return test, test_labels_to_indices, False
            
#### Eurosat

def load_or_skip_eurosat(path, validation, validation_threshold=1600):
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

    for key in list(test.keys()):
        if key not in selected_keys:
            del test[key]
        else:
            #print(test[key].shape)
            test[key] = test[key][-28*4:] # only last 4 layers
    #new_test = {key: test[key] for key in selected_keys}
    #test = new_test
    #del new_test
    
    #load ground_truth once and for all
    with open(path + '/test_classes.pkl', 'rb') as f:
        test_labels_to_indices = pickle.load(f)
    
    return test, test_labels_to_indices, False


#### scores ######################################


def record_head_performance_base(class_activations, cur_activation, label, success_count, hp1_useless, hp2_useless):
    """
    class_activations: (num_sample, num_head, hidden_dim)
    cur_activation: (num_head, hidden_dim)
    success_count is dynamically updated (expected shape: (num_head,))
    """
    # Compute cosine similarities for all samples and heads at once.
    # cur_activation.unsqueeze(0) broadcasts over the sample dimension.
    scores = F.cosine_similarity(class_activations, cur_activation.unsqueeze(0), dim=-1)  # shape: (num_sample, num_head)

    # For each head (along dimension 1), find the sample index with the maximum cosine similarity.
    max_indices = scores.argmax(dim=0)  # shape: (num_head,)

    # Update success_count: increment for heads where the max index equals the label.
    success_count = np.array(success_count)
    success_count[max_indices == label] += 1

    return success_count

def record_head_performance_polar(class_activations, cur_activation, label, success_count, hp1_useless, hp2_useless):
    all_sample = []
    num_classes = class_activations.shape[0]
    num_classes = class_activations.shape[0]

    # Compute cosine similarity for all classes and indices.
    # We unsqueeze cur_activation so that it broadcasts along the num_classes dimension.
    cos_sim = F.cosine_similarity(class_activations, cur_activation.unsqueeze(0), dim=-1)  # shape: (num_classes, n)

    # Vectorized computation of adjusted scores.
    # For each sample (i.e. each column), total is the sum of scores over all classes.
    total = cos_sim.sum(dim=0, keepdim=True)  # shape: (1, n)
    scores_hat = (num_classes * cos_sim - total) / (num_classes - 1)  # shape: (num_classes, n)

    # For each sample, pick the class with the highest adjusted score.
    all_sample = scores_hat.argmax(dim=0)  # shape: (n,)

    all_sample = all_sample.tolist()
    for idx in range(len(all_sample)):
        if all_sample[idx] == label:
            success_count[idx] += 1

    return success_count


def score_arctanh(x, y, alpha=0.45, beta=0.3):
        """score with arctanh"""
        score = ((1 + torch.atanh(x))  * alpha - torch.atanh(y) * beta)
        return score

def record_head_performance_artanh(class_activations, cur_activation, label, success_count, alpha, beta):
    """
    Vectorized version of record_head_performance_artanh.

    Args:
        class_activations (torch.Tensor): Tensor of shape (num_classes, num_heads, hidden_dim).
        cur_activation (torch.Tensor): Tensor of shape (num_heads, hidden_dim).
        label (int): The target class label.
        success_count (torch.Tensor): Tensor of shape (num_heads,) that holds the current success counts.
        alpha (float): The alpha parameter for scoring.
        beta (float): The beta parameter for scoring.

    Returns:
        torch.Tensor: Updated success_count tensor.
    """
    num_classes = class_activations.shape[0]

    # Compute cosine similarity for all classes and heads at once.
    # cur_activation.unsqueeze(0) gives shape (1, num_heads, hidden_dim) which broadcasts over classes.
    scores = F.cosine_similarity(class_activations, cur_activation.unsqueeze(0), dim=-1)  
    # scores has shape (num_classes, num_heads).

    # For each head, compute the total over classes.
    total = scores.sum(dim=0)  # shape: (num_heads,)

    # Compute the mean of the other classes for each class and head.
    # This uses broadcasting: total (shape (num_heads,)) is subtracted from each row of scores.
    y = (total - scores) / (num_classes - 1)  # shape: (num_classes, num_heads)

    # Compute the adjusted score using the arctanh scoring function.
    scores_hat = score_arctanh(scores, y, alpha, beta)  # shape: (num_classes, num_heads)

    # For each head, select the class with the highest adjusted score.
    predictions = scores_hat.argmax(dim=0)  # shape: (num_heads,)

    # Update the success_count vector where the prediction equals the target label.
    # Assumes success_count is a torch.Tensor.
    success_count = np.array(success_count)
    success_count[predictions == label] += 1
    
    return success_count

def record_head_performance_l2(class_activations, cur_activation, label, success_count, hp1_useless, hp2_useless):
    # Compute Euclidean distances between each sample's activation and the current activation for each head.
    # cur_activation.unsqueeze(0) broadcasts over the sample dimension.
    distances = torch.norm(class_activations - cur_activation.unsqueeze(0), p=2, dim=-1)  # shape: (num_sample, num_head)
    
    # Convert distances to similarity scores (smaller distances yield higher scores)
    scores = -distances  # shape: (num_sample, num_head)
    
    # For each head, select the sample index with the maximum similarity (i.e., the smallest Euclidean distance)
    max_indices = scores.argmax(dim=0)  # shape: (num_head,)
    
    # Update success_count for heads where the best matching sample's index equals the label.
    success_count = np.array(success_count)
    success_count[max_indices == label] += 1
    
    return success_count

def record_head_performance_gaussian(class_activations, cur_activation, label, success_count, hp1=1, hp2_useless=0):
    """
    Compute similarity using the Gaussian (RBF) kernel:
        k(v,u) = exp(-hp1 * ||v-u||^2)
    
    Args:
        class_activations: Tensor of shape (num_sample, num_head, hidden_dim)
        cur_activation: Tensor of shape (num_head, hidden_dim)
        label: Target sample index (int) for checking performance.
        success_count: Array-like, tracking success counts per head.
        hp1: Hyperparameter gamma for the Gaussian kernel.
        hp2: Unused for Gaussian kernel.
    
    Returns:
        Updated success_count as a numpy array.
    """
    # Compute squared Euclidean distances between each sample's activation and the current activation.
    diff = class_activations - cur_activation.unsqueeze(0)  # (num_sample, num_head, hidden_dim)
    sq_distances = torch.sum(diff ** 2, dim=-1)               # (num_sample, num_head)
    
    # Apply the Gaussian kernel.
    scores = torch.exp(-hp1 * sq_distances)                   # (num_sample, num_head)
    
    # For each head, select the sample index with the maximum similarity.
    max_indices = scores.argmax(dim=0)                        # (num_head,)
    
    # Update success_count for heads where the selected sample index equals the label.
    success_count = np.array(success_count)
    success_count[max_indices == label] += 1
    
    return success_count

def record_head_performance_laplacian(class_activations, cur_activation, label, success_count, hp1=1, hp2_useless=0):
    """
    Compute similarity using the Laplacian kernel:
        k(v,u) = exp(-hp1 * ||v-u||_1)
    
    Args:
        class_activations: Tensor of shape (num_sample, num_head, hidden_dim)
        cur_activation: Tensor of shape (num_head, hidden_dim)
        label: Target sample index (int) for checking performance.
        success_count: Array-like, tracking success counts per head.
        hp1: Hyperparameter gamma for the Laplacian kernel.
        hp2: Unused for Laplacian kernel.
    
    Returns:
        Updated success_count as a numpy array.
    """
    # Compute L1 distances between each sample's activation and the current activation.
    diff = class_activations - cur_activation.unsqueeze(0)  # (num_sample, num_head, hidden_dim)
    l1_distances = torch.sum(torch.abs(diff), dim=-1)         # (num_sample, num_head)
    
    # Apply the Laplacian kernel.
    scores = torch.exp(-hp1 * l1_distances)                   # (num_sample, num_head)
    
    # For each head, select the sample index with the maximum similarity.
    max_indices = scores.argmax(dim=0)                        # (num_head,)
    
    # Update success_count for heads where the selected sample index equals the label.
    success_count = np.array(success_count)
    success_count[max_indices == label] += 1
    
    return success_count

def record_head_performance_sigmoid(class_activations, cur_activation, label, success_count, hp1=1, hp2=0):
    """
    Compute similarity using the Sigmoid kernel:
        k(v,u) = tanh(hp1 * (v^T u) + hp2)
    
    Args:
        class_activations: Tensor of shape (num_sample, num_head, hidden_dim)
        cur_activation: Tensor of shape (num_head, hidden_dim)
        label: Target sample index (int) for checking performance.
        success_count: Array-like, tracking success counts per head.
        hp1: Hyperparameter alpha for the Sigmoid kernel.
        hp2: Hyperparameter constant c for the Sigmoid kernel.
    
    Returns:
        Updated success_count as a numpy array.
    """
    # Compute dot products between each sample's activation and the current activation.
    # Broadcasting cur_activation to (num_sample, num_head, hidden_dim).
    dot_products = torch.sum(class_activations * cur_activation.unsqueeze(0), dim=-1)  # (num_sample, num_head)
    
    # Apply the Sigmoid kernel.
    scores = torch.tanh(hp1 * dot_products + hp2)           # (num_sample, num_head)
    
    # For each head, select the sample index with the maximum similarity.
    max_indices = scores.argmax(dim=0)                      # (num_head,)
    
    # Update success_count for heads where the selected sample index equals the label.
    success_count = np.array(success_count)
    success_count[max_indices == label] += 1
    
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
    elif score == 'l2':
        record_head_performance = record_head_performance_l2
    elif score == 'gaussian':
        record_head_performance = record_head_performance_gaussian
    elif score == 'laplacian':
        record_head_performance = record_head_performance_laplacian
    elif score == 'sigmoid':
        record_head_performance = record_head_performance_sigmoid
    
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

############# Utils for the Lasso experiments

def get_indices_to_labels(labels_to_indices):
    indices_to_labels = dict()
    for key, values in labels_to_indices.items():
        for value in values:
            indices_to_labels[value] = key
    return indices_to_labels

def act_dict_to_array(d):
    """
    converts a dict of torch tensors to a numpy array of shape (num_samples_dim)
    returns:
         - X : the array (numpy)
         - tensor_shapes : the shape information of the original tensors
         - indices_correspondance : numpy array containing the indices of the 
                    corresponding dict keys (to not lose this information)
    """
    tensor_shapes = d[list(d.keys())[0]].shape #shape of first tensor
    n_samples = len(d)
    total_dim = tensor_shapes[0] * tensor_shapes[1] # n_heads * dim
    X = np.zeros((n_samples, total_dim))
    indices_correspondence = np.zeros(n_samples) #array that will hold the true indices of tensors

    for row_index, (tensor_index, tensor) in enumerate(d.items()):
        indices_correspondence[row_index] = tensor_index
        X[row_index] = d[tensor_index].float().flatten().numpy()

    return X, tensor_shapes, indices_correspondence

def retain_L_last_layers(X, L=2):
    """for both 7B models, we have 28 heads of dim 128 per layer"""
    if L is None: # condition to skip in order to facilitate gridsearch
        return X
    n_last_features = 128 * 28 * L
    return X[:, - n_last_features:]

def get_y(indices_correspondence, indices_to_labels):
    """returns y associated to X, keeping the labels as strings (to use before a label encoder)"""
    y = []
    for i, tensor_ix in enumerate(indices_correspondence):
        y.append(indices_to_labels[tensor_ix])
    return np.array(y)

def evaluate_sklearn(clf, X_val, y_val):
    y_val_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    std = (y_val == y_val_pred).std()
    return acc, std

def evaluate_sklearn_natural_ret(clf, X_val, y_val):

    y_val_pred = clf.predict(X_val)
    results = (y_val_pred == y_val).astype(int)
    # Reshape results into groups of 4 (assumes len(results) is a multiple of 4)
    group_results = results.reshape(-1, 4)
    total_groups = group_results.shape[0]
    # Raw accuracy: average correctness across all samples
    raw_acc = results.mean()
    raw_std = results.std()
    # Question accuracy:
    # For each group:
    #   - The first question is considered correct if both samples 0 and 1 are correct.
    #   - The second question is correct if both samples 2 and 3 are correct.
    q_first = group_results[:, 0] * group_results[:, 1]
    q_second = group_results[:, 2] * group_results[:, 3]
    q_correct = np.sum(q_first) + np.sum(q_second)
    q_acc = q_correct / (total_groups * 2)  # Two questions per group
    q_outcomes = np.concatenate((q_first, q_second))
    q_std = q_outcomes.std()
    # Image accuracy:
    # For each group:
    #   - The first image is correct if samples 0 and 2 are correct.
    #   - The second image is correct if samples 1 and 3 are correct.
    i_first = group_results[:, 0] * group_results[:, 2]
    i_second = group_results[:, 1] * group_results[:, 3]
    i_correct = np.sum(i_first) + np.sum(i_second)
    i_acc = i_correct / (total_groups * 2)  # Two images per group
    i_outcomes = np.concatenate((i_first, i_second))
    i_std = i_outcomes.std()
    # Group accuracy: a group is correct if all four predictions are correct.
    g_outcomes = np.all(group_results == 1, axis=1).astype(float)
    g_acc = g_outcomes.mean()
    g_std = g_outcomes.std()
    
    return raw_acc, raw_std, q_acc, q_std, i_acc, i_std, g_acc, g_std

def fit_predict_lasso(X_train, y_train, hp1, hp2_useless):
    # Train a logistic regression model
    clf = LogisticRegression(solver='liblinear', C=hp1, penalty='l1')
    clf.fit(X_train, y_train)
    return clf

def fit_predict_block_lasso(X_train, y_train, hp1, hp2_useless):
    # Define block structure:
    n_features = X_train.shape[1]
    block_size = 128  # based on heads
    n_groups = n_features // block_size
    # Create a 1D array with group assignments
    group_array = np.repeat(np.arange(n_groups), block_size)
    # If there are remaining features, assign them to an additional group.
    if n_features % block_size:
        group_array = np.concatenate([group_array, 
                                      np.full(n_features % block_size, n_groups)])
    # Convert the 1D group_array into a list of arrays, each containing the indices for that group.
    groups = [np.where(group_array == g)[0] for g in np.unique(group_array)]

    # Instantiate a Logistic Regression estimator with Sparse Group Lasso penalty.
    clf = LogisticSGL(l1_ratio=.5, alpha=hp1, groups=groups, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def match_clf_function(key):
    if key == "lasso":
        return fit_predict_lasso
    elif key == "block_lasso":
        return fit_predict_block_lasso
    else:
        raise ValueError("Unimplemented method")
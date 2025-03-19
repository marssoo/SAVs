import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import pandas as pd
from utils_analysis import *

np.random.seed(0)
validation = True
validation_threshold = 1600

activation_paths = '../activations/'

#Grid parameter
num_heads_grid = [3, 5, 10, 15, 20, 25, 30, 40, 50]
scores_grid = ['base', 'polar', 'artanh']
#for hyper parameters : 
artanh_grid = np.linspace(0.3, 1.5, 6)
empty_grid = [None, None]

# intiate dict of dicts for the results (keys : model\tnum_heads\tscore\thyper_param1\thyper_param2, values : {benchmark : [acc, std]})
result_dict = {}

# go through all benchmark and models
for folder in tqdm(os.listdir(activation_paths)):
    benchmark, model = folder.split('+')
    path = os.path.join(activation_paths, folder)

    #We start by loading the test because we might skip
    #Load the test and val data
    test, test_labels_to_indices, skipping = load_or_skip(path, validation, validation_threshold)
    if skipping:
        print(f'Skipping {benchmark} because too small for validation')
        continue
    #Load the train
    with open(path + '/train_activations.pkl', 'rb') as f:
        activations = pickle.load(f)

    with open(path + '/train_classes.pkl', 'rb') as f:
        labels_to_indices = pickle.load(f)

    #get reverse dictionnary (index to class)
    indices_to_labels = dict()
    for key, values in labels_to_indices.items():
        for value in values:
            indices_to_labels[value] = key

    # get centroids
    centroids = dict()
    for label in labels_to_indices.keys():
        centroids[label] = torch.zeros_like(activations[0]) #initialize null

        # go through associated indices (list of indices)
        for i in labels_to_indices[label]:
            centroids[label] += activations[i]
        #average
        centroids[label] /= len(labels_to_indices[label])
    
    int_to_str = dict()
    str_to_int = dict()
    # shape [num_classes, num_heads, dim_heads]
    class_activations = torch.zeros([len(centroids)] + list(activations[0].shape))

    for i, v in enumerate(centroids.keys()):
        int_to_str[i] = v
        str_to_int[v] = i
        class_activations[i] = centroids[v]

    
    for score in scores_grid:
        if score == 'artanh':
            hyper_param_grid_1,  hyper_param_grid_2 = artanh_grid, artanh_grid
        else:
            hyper_param_grid_1,  hyper_param_grid_2 = empty_grid, empty_grid
        
        for hp1 in hyper_param_grid_1:
            for hp2 in hyper_param_grid_2:
                
                #get success counts
                #print(benchmark, len(test))
                success_counts = get_success_counts(class_activations, activations, str_to_int, indices_to_labels, score, hp1, hp2)

                #try different number of heads
                for num_heads in num_heads_grid:
                    #set key for result dictionnary
                    result_key = '\t'.join([str(e) for e in [model, num_heads, score, hp1, hp2]])
                    #print(f'Running {result_key} on {benchmark}')
                    if result_key not in result_dict.keys():
                        result_dict[result_key] = {} #init dict if not done yet

                    topk_indices = np.sort(np.argsort(success_counts)[-num_heads:][::-1])

                    top_class_activations = get_top_heads(class_activations, topk_indices)

                    if benchmark == 'natural_ret':
                        all_results = evaluate_natural_ret(test, benchmark, activations, topk_indices, top_class_activations, int_to_str, test_labels_to_indices)
                        raw_acc, raw_std, q_acc, q_std, i_acc, i_std, g_acc, g_std = all_results
                        result_dict[result_key][benchmark + '_raw_acc'] = raw_acc
                        result_dict[result_key][benchmark + '_raw_std'] = raw_std
                        result_dict[result_key][benchmark + '_q_acc'] = q_acc
                        result_dict[result_key][benchmark + '_q_std'] = q_std
                        result_dict[result_key][benchmark + '_i_acc'] = i_acc
                        result_dict[result_key][benchmark + '_i_std'] = i_std
                        result_dict[result_key][benchmark + '_g_acc'] = g_acc
                        result_dict[result_key][benchmark + '_g_std'] = g_std
                    else:
                        acc, std = evaluate(test, benchmark, activations, topk_indices, top_class_activations, int_to_str, test_labels_to_indices)
                        result_dict[result_key][benchmark + '_acc'] = acc
                        result_dict[result_key][benchmark + '_std'] = std

columns_str = ['model', 'num_heads', 'score', 'hp1', 'hp2']
n_col = len(columns_str)
new_dict = {}

for index, (key, value) in enumerate(result_dict.items()):
    #get hyper params
    new_dict[index] = {}
    columns = key.split('\t')
    for i in range(n_col):
        new_dict[index][columns_str[i]] = columns[i]
        new_dict[index].update(value)

print(new_dict)
df = pd.DataFrame.from_dict(new_dict, orient='index')
df.to_csv('all_results.csv', sep=',')
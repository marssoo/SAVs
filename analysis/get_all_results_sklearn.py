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
validation_threshold = 1000

activation_paths = '../activations/'

#Grid parameter
Cs = [10 ** k for k in range(-5, 3)]
#scores_grid = ['lasso']
scores_grid = ['block_lasso']

n_last_layers = [1, 2, 3, 4]
empty_grid = [None]

# intiate dict of dicts for the results (keys : model\tnum_heads\tscore\thyper_param1\thyper_param2, values : {benchmark : [acc, std]})
result_dict = {}
num_heads = None

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
        train = pickle.load(f)

    with open(path + '/train_classes.pkl', 'rb') as f:
        train_labels_to_indices = pickle.load(f)

    #reverse truth tables
    train_indices_to_labels = get_indices_to_labels(train_labels_to_indices)
    val_indices_to_labels = get_indices_to_labels(test_labels_to_indices)

    X_train, tensor_shapes, train_ix = act_dict_to_array(train)
    X_val, _, val_ix = act_dict_to_array(test)
    y_train, y_val = get_y(train_ix, train_indices_to_labels), get_y(val_ix, val_indices_to_labels)
    # label encode
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    for score in scores_grid:
        fit_predict = match_clf_function(score) #here just lasso

        for hp1 in Cs:
            for hp2 in n_last_layers:
                result_key = '\t'.join([str(e) for e in [model, num_heads, score, hp1, hp2]]) # for tracking

                X_train_cut = retain_L_last_layers(X_train, hp2)
                X_val_cut = retain_L_last_layers(X_val, hp2)

                clf = fit_predict(X_train_cut, y_train_enc, hp1, hp2) #hp1 is the regularizer C for lasso
                
                print(f'Running {result_key} on {benchmark}')
                if result_key not in result_dict.keys():
                    result_dict[result_key] = {} #init dict if not done yet


                if benchmark == 'natural_ret':
                    all_results = evaluate_sklearn_natural_ret(clf, X_val_cut,  y_val_enc)
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
                    acc, std = evaluate_sklearn(clf, X_val_cut, y_val_enc)
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

#print(new_dict)
df = pd.DataFrame.from_dict(new_dict, orient='index')
df.to_csv('result_csvs/block_lasso_val_last_layers.csv', sep=',', index=False)
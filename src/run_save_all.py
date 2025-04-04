from .utils import *
#from .utils_modif_0 import *

from .model import *
from .preprocess import *
from tqdm import tqdm
import torch
import argparse
torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error() 

import os
import pickle
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)



def eval_dataset(args):
    model = load_model(args.model_name, args.data_name, quantize=args.quantize)

    train_data = open_data(args.data_name, args.train_path)
    test_data = open_data(args.data_name, args.val_path)
    
    # create dump directories if they don't already exist
    pth = 'activations'
    if not os.path.exists(pth):
        os.makedirs(pth)

    if args.data_name == 'blink':
        splits = args.train_path.split('\.')[0]
        name = splits.split('/')[-1]
        name = name.split('_')[:-1]
        name = '_'.join(name)
        sub_dir = f'{pth}/{name}_{args.model_name}/'
    else:
        sub_dir = f'{pth}/{args.data_name}_{args.model_name}/'

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


    #train :
    acts, labels_to_indices = get_activations(train_data, model, model.all_heads)
    #put on cpu before pickling
    for act in acts:
        acts[act] = acts[act].cpu()

    with open(sub_dir + "train_activations.pkl", "wb") as f:
        pickle.dump(acts, f)
    with open(sub_dir + "train_classes.pkl", "wb") as f:
        pickle.dump(labels_to_indices, f)

    

    test_labels_to_indices = dict()
    for i in tqdm(range(0, len(test_data), 1000)):
        chunk = test_data[i:i+1000]

        acts, chunks_labels_to_indices = get_activations(chunk, model, model.all_heads)
        #adjust indices 
        new_acts = dict()
        for index in acts.keys():
            new_acts[index + i] = acts[index]
        acts = new_acts
        for label in chunks_labels_to_indices:
            chunks_labels_to_indices[label] = [x + i for x in chunks_labels_to_indices[label]]
        #put on cpu
        for act in acts:
            acts[act] = acts[act].cpu()
        #save chunk's acts
        with open(sub_dir + f"test_activations_{int(i/1000)}.pkl", "wb") as f:
            pickle.dump(acts, f)
        #update label dict
        for key, value in chunks_labels_to_indices.items():
            test_labels_to_indices.setdefault(key, []).extend(value)
    #save classes
    with open(sub_dir + "test_classes.pkl", "wb") as f:
        pickle.dump(test_labels_to_indices, f)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava_ov_0.5b")
    parser.add_argument("--data_name", type=str, default="Mhalu")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--eval_zeroshot", action="store_true", 
                       help="Whether to run zero-shot evaluation")
    #new arg
    parser.add_argument('--quantize', type=int, default=None,
                        help="number of bits to operate quantization on.")
    args = parser.parse_args()
    print(f'\n######## Running {args.model_name} on {args.data_name} ########\n')
    eval_dataset(args)
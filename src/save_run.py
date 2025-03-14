from .utils import *
from .saving_utils import *
from .model import *
from .preprocess import *
from tqdm import tqdm
import torch
import argparse
torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error() 



def eval_dataset(args):
    model = load_model(args.model_name, args.data_name)

    train_data = open_data(args.data_name, args.train_path)
    test_data = open_data(args.data_name, args.val_path)

    # Zero-shot evaluation
    if args.eval_zeroshot:
        zs_correct = 0
        for item in tqdm(test_data, desc="Zero-shot Evaluation"):
            # Create model input
            model_input = model.insert_image(item['question'], [item['image']])
            
            # Generate response and clean it
            output = model.generate(model_input, max_new_tokens=32).strip()
            
            # Direct comparison with dataset label
            if output == item['label']:
                zs_correct += 1
                
        print("Zero-shot Accuracy:", zs_correct / len(test_data))
    else:
        # SAVs embeddings
        multimodal_embeddings = save_mllm_encode(model, train_data, num_head=args.num_head)
         
        int_to_str = multimodal_embeddings['int_to_str']
        top_heads = multimodal_embeddings['top_heads']
        class_activations =  multimodal_embeddings["activations"]

        save_top_heads(top_heads, int_to_str, class_activations, args.file_path)
        print(f"Top heads saved to {args.file_path}")


        correct_count = 0
        ###Checking how well it can classify a given query
        for item in tqdm(test_data):
            cur_class = mllm_classify(item, model, multimodal_embeddings)
            if item['label'] == cur_class:
                correct_count += 1
        print("SAVs Accuracy:", correct_count / len(test_data))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava_ov")
    parser.add_argument("--num_head", type=int, default=20)
    parser.add_argument("--data_name", type=str, default="Mhalu")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--eval_zeroshot", action="store_true", 
                       help="Whether to run zero-shot evaluation")
    parser.add_argument("--file_path", type=str, required=True, 
                        help="File path to save the top heads")
    
    args = parser.parse_args()
    eval_dataset(args)
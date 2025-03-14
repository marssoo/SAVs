from .utils_per_class import *
from .model import *
from .preprocess import *
from tqdm import tqdm
import torch
import argparse
from collections import defaultdict
torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error()

def eval_dataset(args):
    # Load the model
    model = load_model(args.model_name, args.data_name)
    train_data = open_data(args.data_name, args.train_path)
    test_data = open_data(args.data_name, args.val_path)

    if args.eval_zeroshot:
        # Zero-shot evaluation remains the same
        zs_correct = 0
        for item in tqdm(test_data, desc="Zero-shot Evaluation"):
            model_input = model.insert_image(item['question'], [item['image']])
            output = model.generate(model_input, max_new_tokens=32).strip()
            if output == item['label']:
                zs_correct += 1
        print("Zero-shot Accuracy:", zs_correct / len(test_data))
    else:
        # Class-specific head selection
        print("Encoding class-specific heads...")
        class_embeddings = mllm_encode_per_class(
            model, 
            train_data,
            num_heads_per_class=args.num_head
        )
        
        # Save class embeddings
        save_class_embeddings(class_embeddings, args.file_path)
        print(f"Class embeddings saved to {args.file_path}")

        # Class-specific voting evaluation
        correct_count = 0
        for item in tqdm(test_data, desc="Class-specific Evaluation"):
            try:
                # Format input for the model
                model_input = model.format_func(None, item, num_shot=0, model_helper=model)
                inputs = model.insert_image(model_input[0], model_input[1])
                
                # Get prediction using class-specific voting
                prediction = mllm_classify_per_class(inputs, model, class_embeddings)
                if prediction == item['label']:
                    correct_count += 1
            except Exception as e:
                print(f"Error processing item: {e}")
                continue

        print(f"Class-specific SAVs Accuracy: {correct_count / len(test_data):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava_ov")
    parser.add_argument("--num_head", type=int, default=20,
                       help="Number of heads per class")
    parser.add_argument("--data_name", type=str, default="Mhalu")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--eval_zeroshot", action="store_true",
                       help="Run zero-shot evaluation instead of SAVs")
    parser.add_argument("--file_path", type=str, required=True,
                       help="Path to save/load class embeddings")

    args = parser.parse_args()
    eval_dataset(args)
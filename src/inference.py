from .utils import *
from .model import *
from .preprocess import *
from tqdm import tqdm
import torch
import argparse
import json
import os
torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error()

def run_inference(args):
    """
    Run inference on a dataset using a pre-trained model and saved top heads.

    Parameters:
    - args: Command-line arguments containing model_name, saved_heads_path, data_path, and output_path.
    """
    # Load the model
    model = load_model(args.model_name, args.data_name)

    # Load the saved top heads and label mappings
    class_embed = load_top_heads(args.saved_heads_path)
    top_heads = class_embed['top_heads']
    int_to_str = class_embed['int_to_str']

    # Load the input data (JSONL file)
    test_data = open_data(args.data_name, args.data_path)

    # Recompute activations for the saved top heads
    print("Recomputing activations for the saved top heads...")
    class_activations, _, _ = get_class_activations(test_data, model, top_heads)
    class_embed['activations'] = class_activations  # Add activations to class_embed

    # Run inference
    results = []
    correct_count = 0

    for item in tqdm(test_data, desc="Running Inference"):
        # Get the model prediction
        cur_class = mllm_classify(item, model, class_embed)

        # Save the result
        result = {
            "input": item,
            "output": cur_class
        }
        results.append(result)

        # Check if the prediction is correct
        if cur_class == item['label']:
            correct_count += 1

    # Compute accuracy
    accuracy = correct_count / len(test_data)
    print(f"Accuracy: {accuracy:.4f}")

    # Save the results to the output file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Inference results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a pre-trained model and saved top heads.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load.")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--saved_heads_path", type=str, required=True, help="Path to the saved top heads and label mappings.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL file containing the input data.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the inference results.")

    args = parser.parse_args()

    # Run inference
    run_inference(args)
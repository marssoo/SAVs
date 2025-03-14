from .utils_per_class import *
from .model import *
from .preprocess import *
from tqdm import tqdm
import torch
import argparse
import json
import os
from collections import defaultdict
torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error()

def run_inference(args):
    """
    Run inference using class-specific heads and voting mechanism.
    """
    # Load model and class embeddings
    model = load_model(args.model_name, args.data_name)
    class_embeddings = load_class_embeddings(args.saved_heads_path)
    
    # Extract shared label mapping
    int_to_str = class_embeddings.pop('int_to_str')
    class_labels = [cls for cls in class_embeddings if cls != 'int_to_str']

    # Load test data
    test_data = open_data(args.data_name, args.data_path)

    # Collect all unique heads across all classes
    all_heads = []
    for cls, embedding in class_embeddings.items():
        all_heads.extend(embedding['top_heads'])
    unique_heads = list(set(all_heads))

    # Precompute activations for all relevant heads
    print("Precomputing activations...")
    query_activations = precompute_query_activations(test_data, model, unique_heads)

    # Create head index mapping
    head_to_idx = {head: idx for idx, head in enumerate(unique_heads)}

    # Prepare class data with indices and tensors
    for cls, embedding in class_embeddings.items():
        # Convert stored lists to tensors
        embedding['activations'] = torch.stack([
            torch.tensor(embedding['activations'][str(i)]) 
            for i in range(len(embedding['top_heads']))
        ])
        
        # Map head tuples to precomputed indices
        embedding['head_indices'] = [head_to_idx[tuple(h)] for h in embedding['top_heads']]

    # Run inference
    results = []
    correct_count = 0
    
    for idx, item in enumerate(tqdm(test_data, desc="Processing items")):
        item_act = query_activations[idx]
        votes = defaultdict(int)

        # Vectorized similarity calculation
        for cls, emb in class_embeddings.items():
            # Get relevant activations for this class
            cls_act = item_act[emb['head_indices']]
            
            # Calculate similarities for all heads at once
            similarities = F.cosine_similarity(
                emb['activations'].unsqueeze(1),  # (C, 1, D)
                cls_act.unsqueeze(0),             # (1, K, D)
                dim=-1
            )  # (C, K)
            
            # Count votes where max similarity matches class index
            class_idx = int([k for k, v in int_to_str.items() if v == cls][0])
            votes[cls] += (similarities.argmax(0) == class_idx).sum().item()

        # Determine final prediction
        prediction = max(votes, key=votes.get) if votes else "unknown"
        results.append({
            "input_id": idx,
            "prediction": prediction,
            "ground_truth": item['label'],
            "votes": dict(votes)
        })

        if prediction == item['label']:
            correct_count += 1

    # Save results and calculate accuracy
    accuracy = correct_count / len(test_data)
    print(f"Class-specific Voting Accuracy: {accuracy:.4f}")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "predictions": results,
            "label_mapping": int_to_str
        }, f, indent=4)

def precompute_query_activations(data, model, heads):
    """Precompute activations for all specified heads with correct dimensions"""
    # Get model configuration parameters
    n_heads = model.model_config['n_heads']
    resid_dim = model.model_config['resid_dim']
    head_dim = resid_dim // n_heads  # Actual dimension per attention head

    # Initialize tensor with correct dimensions
    activations = torch.zeros(len(data), len(heads), head_dim)
    
    for idx, item in enumerate(tqdm(data, desc="Precomputing")):
        mean_act = get_last_mean_head_activations([item], model, N_TRIALS=1, shot=0)
        
        # Extract activations for each head with proper dimension handling
        head_activations = []
        for head in heads:
            layer_idx, head_idx = head[0], head[1]
            activation = mean_act[layer_idx, head_idx, -1]  # (head_dim,)
            head_activations.append(activation)
        
        activations[idx] = torch.stack(head_activations)
    
    return activations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-specific SAVs Inference")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--saved_heads_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    run_inference(args)
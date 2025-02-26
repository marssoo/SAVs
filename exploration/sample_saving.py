import json
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def visualize_samples(jsonl_path, num_samples):
    """
    Visualizes randomly selected samples from a JSONL file.

    Args:
        jsonl_path (str): Path to the JSONL file.
        num_samples (int): Number of samples to visualize.
    """
    data = []
    with open(jsonl_path, "r") as file:
        for line in file:
            data.append(json.loads(line.strip()))

    num_samples = min(num_samples, len(data))
    selected_samples = random.sample(data, num_samples)

    for idx, sample in enumerate(selected_samples):
        image_path = sample.get("image", "")
        question = sample.get("question", "No question available")
        label = sample.get("label", "No label")

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            continue

        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Q: {question}\nLabel: {label}", fontsize=10, wrap=True)
        plt.tight_layout(pad=2.0)
        save_path = f"sample_plot_{idx}.png"
        plt.savefig(save_path)  
        print(f"Plot saved as {save_path}")

        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize random samples from a JSONL dataset.")
    parser.add_argument("jsonl_path", type=str, help="Path to the JSONL file.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize.")
    
    args = parser.parse_args()
    
    visualize_samples(args.jsonl_path, args.num_samples)

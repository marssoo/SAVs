import os
import json
import itertools
import re
import random

# Define the data folder
data_folder = "comp_data"
output_full = "comp_pairs.jsonl"
output_train = "comp_train.jsonl"
output_val = "comp_val.jsonl"

# Regex to extract the car name (before the number)
def extract_base_name(filename):
    match = re.match(r"([a-zA-Z]+)", filename)  # Extracts only the alphabetic prefix
    return match.group(1) if match else None

# Get all images in the folder
image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Create pairs without repetition
pairs = list(itertools.combinations(image_files, 2))

# Generate JSONL data and separate into "Yes" and "No"
yes_pairs = []
no_pairs = []
for question_id, (img1, img2) in enumerate(pairs, start=1):
    base1, base2 = extract_base_name(img1), extract_base_name(img2)
    label = "Yes" if base1 == base2 else "No"
    
    entry = {
        "question_id": question_id,
        "image_1": os.path.join(data_folder, img1),
        "image_2": os.path.join(data_folder, img2),
        "question": "Do these images depict the same car, considering they may be captured from different angles, lighting conditions, or distances?",
        "label": label
    }
    
    if label == "Yes":
        yes_pairs.append(entry)
    else:
        no_pairs.append(entry)

# Shuffle "Yes" and "No" pairs before creating the train dataset
random.shuffle(yes_pairs)
random.shuffle(no_pairs)

# Select 20 samples from both "Yes" and "No" pairs to create a total of 40 training samples
train_data = yes_pairs[:20] + no_pairs[:20]

# Shuffle the train data before saving
random.shuffle(train_data)

# Combine remaining samples for validation
val_data = yes_pairs[20:] + no_pairs[20:]

# Save full dataset
with open(output_full, "w") as f:
    for entry in yes_pairs + no_pairs:
        f.write(json.dumps(entry) + "\n")

# Save train dataset
with open(output_train, "w") as f:
    for entry in train_data:
        f.write(json.dumps(entry) + "\n")

# Save val dataset
with open(output_val, "w") as f:
    for entry in val_data:
        f.write(json.dumps(entry) + "\n")

print(f"Files created:\n- {output_full} (Full dataset)\n- {output_train} (Train, {len(train_data)} samples)\n- {output_val} (Val, {len(val_data)} samples)")

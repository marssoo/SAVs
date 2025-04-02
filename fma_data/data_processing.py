import os
import json
import random
from itertools import combinations, product

# Path to the folder containing images
DATA_FOLDER = "/home/onyxia/work/fma_data"
OUTPUT_TRAIN = "/home/onyxia/work/SAVs/fma_data/fma_train.jsonl"
OUTPUT_VAL = "/home/onyxia/work/SAVs/fma_data/fma_test.jsonl"

# Read all images in the folder
images = [img for img in os.listdir(DATA_FOLDER) if img.endswith(".png")]

# Categorize images
single_car_images = {}  # {carName: [img1, img2, ...]}
accident_images = {}  # {(accidentNumber, carName): [img1, img2, ...]}

for img in images:
    parts = img.split("_")
    if len(parts) == 1:  # Single car type (e.g., audi1.png)
        car_name = ''.join(filter(str.isalpha, img))  # Extract car name
        single_car_images.setdefault(car_name, []).append(img)
    else:  # Accident images (e.g., 1_kia_02.png)
        accident_number, car_name, _ = parts
        accident_key = (accident_number, car_name)
        accident_images.setdefault(accident_key, []).append(img)

# Generate all possible pairs
all_pairs = []

# 1. Pair within Single-Single
for car_name, img_list in single_car_images.items():
    for img1, img2 in combinations(img_list, 2):
        all_pairs.append((img1, img2, "Yes"))  # Same car

# 2. Pair within Accident-Accident
for (accident_number, car_name), img_list in accident_images.items():
    for img1, img2 in combinations(img_list, 2):
        all_pairs.append((img1, img2, "Yes"))  # Same accident & car

# 3. Pair Single-Accident and Mixed Accidents
single_car_list = [(car, img) for car, imgs in single_car_images.items() for img in imgs]
accident_car_list = [(f"{accident}_{car}", img) for (accident, car), imgs in accident_images.items() for img in imgs]

for (name1, img1), (name2, img2) in product(single_car_list + accident_car_list, repeat=2):
    if img1 >= img2:  # Avoid duplicate pairs
        continue

    # Check label
    if name1 == name2:
        label = "Yes"  # Same car
    else:
        label = "No"  # Different cars or different accidents

    all_pairs.append((img1, img2, label))

# Shuffle all pairs
random.shuffle(all_pairs)

# Create train set (20 Yes, 20 No)
train_yes = [pair for pair in all_pairs if pair[2] == "Yes"][:20]
train_no = [pair for pair in all_pairs if pair[2] == "No"][:20]
train_set = train_yes + train_no

# Put the rest in validation
val_set = [pair for pair in all_pairs if pair not in train_set]

# Helper function to save as JSONL
def save_jsonl(data, filename):
    with open(filename, "w") as f:
        for idx, (img1, img2, label) in enumerate(data):
            json_obj = {
                "question_id": idx,
                "image_1": os.path.join(DATA_FOLDER, img1),
                "image_2": os.path.join(DATA_FOLDER, img2),
                "question": "Do these images depict the same car, considering they may be captured from different angles, lighting conditions, or distances?",
                "label": label
            }
            f.write(json.dumps(json_obj) + "\n")


# Save to files
save_jsonl(train_set, OUTPUT_TRAIN)
save_jsonl(val_set, OUTPUT_VAL)

print(f"Saved {len(train_set)} training examples to {OUTPUT_TRAIN}")
print(f"Saved {len(val_set)} validation examples to {OUTPUT_VAL}")

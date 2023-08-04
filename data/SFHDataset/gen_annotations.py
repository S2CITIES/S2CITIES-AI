import os
import random
import json
import argparse

parser = argparse.ArgumentParser(
    prog = 'Script to generate SignalForHelp dataset annotations.'
)
parser.add_argument('--data_path', default='dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_simplified_ratio1', type=str, help='Path for train/test/val video files.')
args = parser.parse_args()

# Create the annotation files
train_file = 'data/SFHDataset/train_annotations.txt'
val_file = 'data/SFHDataset/val_annotations.txt'
test_file = 'data/SFHDataset/test_annotations.txt'
info_file = 'data/SFHDataset/info.json'
target_dataset = args.data_path.split('/')[-1]

# Set the split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Function to write annotations to a file
def write_annotations(file_path, annotations):
    with open(file_path, 'w') as f:
        for annotation in annotations:
            f.write(annotation + '\n')

# Collect all video paths and labels
video_paths = []
for label in [0, 1]:
    class_folder = os.path.join(args.data_path, str(label))
    for video_file in os.listdir(class_folder):
        video_path = os.path.join(class_folder, video_file)
        video_paths.append((video_path, label))

# Shuffle the video paths
random.shuffle(video_paths)

# Split the dataset
total_samples = len(video_paths)
train_split = int(total_samples * train_ratio)
val_split = int(total_samples * (train_ratio + val_ratio))

train_data = video_paths[:train_split]
val_data = video_paths[train_split:val_split]
test_data = video_paths[val_split:]

# Generate annotations for each split
train_annotations = [f'{video_path} {label}' for video_path, label in train_data]
val_annotations = [f'{video_path} {label}' for video_path, label in val_data]
test_annotations = [f'{video_path} {label}' for video_path, label in test_data]

# Write annotations to files
write_annotations(train_file, train_annotations)
write_annotations(val_file, val_annotations)
write_annotations(test_file, test_annotations)

print("Annotation files generated successfully.")

# Count the number of negatives and positives in each split
def count_labels(data):
    num_negatives = sum(1 for _, label in data if label == 0)
    num_positives = sum(1 for _, label in data if label == 1)
    return num_negatives, num_positives

train_negatives, train_positives = count_labels(train_data)
val_negatives, val_positives = count_labels(val_data)
test_negatives, test_positives = count_labels(test_data)
print('--- Statitics ---')
print("Train Set: Negatives:", train_negatives, "Positives:", train_positives)
print("Validation Set: Negatives:", val_negatives, "Positives:", val_positives)
print("Test Set: Negatives:", test_negatives, "Positives:", test_positives)

with open(info_file, 'r') as file:
    info = json.load(file)

# Init info dictionary
info[target_dataset] = {}
info[target_dataset]['statistics'] = {}
info[target_dataset]['statistics']['train'] = {}
info[target_dataset]['statistics']['test'] = {}
info[target_dataset]['statistics']['val'] = {}

# mean and std will be saved as empty and filled while training (if not already done)
info[target_dataset]['mean'] = []
info[target_dataset]['std'] = []

info[target_dataset]['statistics']['train']['positives'] = train_positives
info[target_dataset]['statistics']['train']['negatives'] = train_negatives
info[target_dataset]['statistics']['test']['positives'] = test_positives
info[target_dataset]['statistics']['test']['negatives'] = test_negatives
info[target_dataset]['statistics']['val']['positives'] = val_positives
info[target_dataset]['statistics']['val']['negatives'] = val_negatives

with open(info_file, 'w') as file:
    json.dump(info, file, indent=4)  # indent=4 for pretty formatting

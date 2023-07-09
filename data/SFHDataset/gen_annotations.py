import os
import random

# Set the paths to your dataset folders
dataset_folder = 'dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_simplified_ratio1_224x224'

# Create the annotation files
train_file = 'data/SFHDataset/train_annotations.txt'
val_file = 'data/SFHDataset/val_annotations.txt'
test_file = 'data/SFHDataset/test_annotations.txt'

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
    class_folder = os.path.join(dataset_folder, str(label))
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


"""
This file is used to extract the keypoints from the videos in the dataset by leveraging the FeatureExtractor class.
"""

import os
from pathlib import Path
import json

import numpy as np

from featureextractor import FeatureExtractor

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

# Read from json file
with open("const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

# Define allowed extensions
allowed_extensions = const["VIDEO_EXTENSIONS"]

# Define paths
input_folder = Path("dataset_creation/4_videos_labeled_subsampled")
output_folder = Path("dataset_creation/5_features_extracted")

# Create output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# Loop through each label folder
for label in ["0", "1"]:
    # Create label output folder if it doesn't exist
    label_output_folder = output_folder / label
    label_output_folder.mkdir(parents=True, exist_ok=True)

    # Loop through each video in the label folder
    for video_file in sorted((input_folder / label).glob("*")):

        # Skip if not a video file
        if not video_file.suffix.lower() in allowed_extensions:
            continue

        # Extract keypoints from the video
        feature_extractor = FeatureExtractor(str(video_file), show_image=True)
        keypoints = feature_extractor.extract_keypoints_from_video()

        # Save keypoints as npy file if keypoints is not None
        if keypoints is not None:
            keypoints_file = video_file.stem + ".npy"
            keypoints_path = label_output_folder / keypoints_file
            np.save(keypoints_path, keypoints)
        else:
            print("No hand detected for the whole video, or multiple hands detected.")
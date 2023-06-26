"""
This file is used to extract the keypoints from the videos in the dataset by leveraging the FeatureExtractor class.
"""

import os
from pathlib import Path
import json

import numpy as np

from featureextractor import KeypointsExtractor

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

# Read from json file
with open("const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

# Define allowed extensions
allowed_extensions = const["VIDEO_EXTENSIONS"]

# Define paths
input_folder = Path("dataset_creation") / const["VIDEOS_LABELED_SUBSAMPLED"]
output_folder = Path("dataset_creation") / const["FEATURES_EXTRACTED"]

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
        
        keypoints_file = video_file.stem + ".npy"
        keypoints_path = label_output_folder / keypoints_file

        # If the keypoints file already exists, skip it
        if keypoints_path.is_file():
            print(f"Skipping {video_file}")
            continue

        # Print the current file
        print(f"Processing {video_file}")
        
        # Extract keypoints from the video
        feature_extractor = KeypointsExtractor(str(video_file), show_image=True)
        keypoints = feature_extractor.extract_keypoints_from_video()

        # Save keypoints as npy file if keypoints is not None
        if keypoints is not None:
            np.save(keypoints_path, keypoints)
        else:
            print("No hand detected for the whole video, or multiple hands detected.")

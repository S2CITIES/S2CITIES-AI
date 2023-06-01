# -*- coding: utf-8 -*-

import pandas as pd
from utils import get_video_files
import json

from pathlib import Path
import os

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

# This function creates a CSV file where each row is a video
# in the folder passed to the function. The columns are the
# video file name, a processed flag, and the label.
# Initialize the processed flag to 0 and the label to -1.
def create_csv(folder, csv_filename):
    # Get a list of all files in the folder
    video_files = get_video_files(folder)

    # Create an empty list to store the video information
    video_info = []

    # Loop through each video file
    for file in video_files:
        # Add the video information to the list
        video_info.append({
            "file": file.name,
            "processed": 0,
            "label": -1
        })

    # Create a dataframe from the video information
    df = pd.DataFrame(video_info)

    # Save the dataframe to a CSV file
    df.to_csv(csv_filename, index=False)

# Read from json file
with open("../const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

# Set the path to the dataset folder
DATASET_FOLDER = Path(const["DATASET_FOLDER"])

create_csv(DATASET_FOLDER, 'dataset.csv')
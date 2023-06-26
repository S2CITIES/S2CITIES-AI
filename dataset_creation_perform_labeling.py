"""
The following is for being used when actually performing the labeling
"""

from pathlib import Path
import os
import json

from src.dataset_creation.videolabeler import VideoLabeler

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

# Read from json file
with open("./src/const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

labeler = VideoLabeler(video_extensions=const["VIDEO_EXTENSIONS"])
labeler.read_dataframe("labeling.csv")
labeler.label_videos("3_videos_splitted")
labeler.update_csv("labeling.csv")

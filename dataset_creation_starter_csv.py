"""
The following is for being used when creating the labeling csv file.
"""

from pathlib import Path
import os
import json

from src.dataset_creation.videolabeler import VideoLabeler

# Read from json file
with open("./src/const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

labeler = VideoLabeler(video_extensions=const["VIDEO_EXTENSIONS"])
labeler.create_starter_csv("3_videos_splitted", "labeling.csv")

"""
The following is for being used when the labeling is already done
and we want to move the files to the corresponding folders
"""

from pathlib import Path
import os
import json

from src.dataset_creation.videolabeler import VideoLabeler

# Read from json file
with open("./src/const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

labeler = VideoLabeler(video_extensions=const["VIDEO_EXTENSIONS"])
labeler.read_dataframe("labeling.csv")
labeler.move_files(source_folder="3_videos_splitted", destination_folder="4_videos_labeled")

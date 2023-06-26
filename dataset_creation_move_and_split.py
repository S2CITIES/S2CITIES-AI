"""
The following is for being used when creating the dataset clips form the collected videos.
"""

from pathlib import Path
import os
import json

from src.dataset_creation.videoprocessor import VideoProcessor

# Read from json file
with open("./src/const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

processor = VideoProcessor(
    videos_arrived_folder=const["VIDEOS_ARRIVED"],
    videos_raw_folder=const["VIDEOS_RAW"],
    videos_raw_processed_folder=const["VIDEOS_RAW_PROCESSED"],
    videos_splitted_folder=const["VIDEOS_SPLITTED"],
    videos_labeled_folder=const["VIDEOS_LABELED"],
    video_extensions=const["VIDEO_EXTENSIONS"],
    subclip_duration=2.5,
    shift_duration=1,
    starting_idx=275
    )

processor.move_arrived_videos()
processor.split_raw_videos()

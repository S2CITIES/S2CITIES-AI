"""
Main script for dataset creation.
"""

from pathlib import Path
import os
import json

# Import classes from other files

from videoprocessor import VideoProcessor
from videolabeler import VideoLabeler

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

# Read from json file
with open("../const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

# The following is for being used when creating the dataset clips form the collected videos

# processor = VideoProcessor(
#     videos_arrived_folder=const["VIDEOS_ARRIVED"],
#     videos_raw_folder=const["VIDEOS_RAW"],
#     videos_raw_processed_folder=const["VIDEOS_RAW_PROCESSED"],
#     videos_splitted_folder=const["VIDEOS_SPLITTED"],
#     videos_labeled_folder=const["VIDEOS_LABELED"],
#     video_extensions=const["VIDEO_EXTENSIONS"],
#     subclip_duration=2.5,
#     shift_duration=1
#     )
# processor.move_arrived_videos()
# processor.split_raw_videos()

# The following is for being used when creating the labeling csv file

# labeler = VideoLabeler(video_extensions=const["VIDEO_EXTENSIONS"])
# labeler.create_starter_csv("3_videos_splitted", "labeling.csv")

# The following is for being used when actually performing the labeling

# labeler = VideoLabeler(video_extensions=const["VIDEO_EXTENSIONS"])
# labeler.read_dataframe("labeling.csv")
# labeler.label_videos("3_videos_splitted")
# labeler.update_csv("labeling.csv")

# The following is for being used when the labeling is already done
# and we want to move the files to the corresponding folders

# labeler = VideoLabeler(video_extensions=const["VIDEO_EXTENSIONS"])
# labeler.read_dataframe("labeling.csv")
# labeler.move_files(source_folder="3_videos_splitted", destination_folder="4_videos_labeled")

"""
Main script for dataset creation.
"""

from pathlib import Path
import os

from videoprocessor import VideoProcessor
from videolabeler import VideoLabeler

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

# The following is for being used when creating the dataset clips form the collected videos

processor = VideoProcessor("../const.json", subclip_duration=2.5, shift_duration=1)
processor.move_arrived_videos()
processor.split_raw_videos()

# The following is for being used when actually performing the labeling

labeler = VideoLabeler("../const.json")
labeler.create_starter_csv("3_videos_splitted", "labeling.csv")
labeler.read_dataframe("labeling.csv")
labeler.label_videos("3_videos_splitted")
labeler.update_csv("labeling.csv")

# The following is for being used when the labeling is already done
# and we want to move the files to the corresponding folders

labeler = VideoLabeler("../const.json")
labeler.read_dataframe("labeling.csv")
labeler.move_files("3_videos_splitted")

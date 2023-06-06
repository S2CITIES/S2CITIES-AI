"""
Main script for dataset creation.
"""

from pathlib import Path
import os

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

from videoprocessor import VideoProcessor

# Create VideoProcessor instance
processor = VideoProcessor("../const.json", subclip_duration=2.5, shift_duration=1)
#processor.cleanup_folder()
#processor.move_arrived_videos()
#processor.split_raw_videos()

from videolabeler import VideoLabeler

#labeler = VideoLabeler("../const.json")
#labeler.create_starter_csv("3_videos_splitted", "labeling.csv")
#labeler.read_dataframe("labeling.csv")
#labeler.label_videos("3_videos_splitted")
#labeler.update_csv("labeling.csv")


labeler = VideoLabeler("../const.json")
labeler.read_dataframe("labeling.csv")
labeler.move_files("3_videos_splitted")
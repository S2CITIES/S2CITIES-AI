"""
The following is for being used when creating the dataset clips form the collected videos.
"""

import argparse
from src.dataset_creation.videoprocessor import VideoProcessor
from src import constants

parser = argparse.ArgumentParser()
parser.add_argument('--starting_idx', type=int,
                    help='Starting index for next video to be processed.')
args = parser.parse_args()

processor = VideoProcessor(
    videos_arrived_folder=constants.VIDEOS_ARRIVED,
    videos_raw_folder=constants.VIDEOS_RAW,
    videos_raw_processed_folder=constants.VIDEOS_RAW_PROCESSED,
    videos_splitted_folder=constants.VIDEOS_SPLITTED,
    videos_labeled_folder=constants.VIDEOS_LABELED,
    video_extensions=constants.VIDEO_EXTENSIONS,
    subclip_duration=constants.SUBCLIP_DURATION,
    shift_duration=constants.SHIFT_DURATION,
    starting_idx=args.starting_idx,
)

processor.move_arrived_videos()
processor.split_raw_videos()

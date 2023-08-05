"""
The following is for being used when creating the dataset clips form the collected videos.
"""

import argparse
from src.dataset_creation.videoprocessor import VideoProcessor
from src import constants

parser = argparse.ArgumentParser()
parser.add_argument('--starting_idx', type=int, required=True,
                    help='Starting index for next video to be processed.')
parser.add_argument('--path_videos_arrived', type=str, required=True,
                    help='Folder containing the arrived videos.')
parser.add_argument('--path_videos_raw', type=str, required=True,
                    help='Folder containing the renamed arrived videos.')
parser.add_argument('--path_videos_raw_processed', type=str, required=True,
                    help='Folder containing the processed videos.')
parser.add_argument('--path_videos_splitted', type=str, required=True,
                    help='Folder containing the splitted videos.')
args = parser.parse_args()

processor = VideoProcessor(
    videos_arrived_folder=args.path_videos_arrived,
    videos_raw_folder=args.path_videos_raw,
    videos_raw_processed_folder=args.path_videos_raw_processed,
    videos_splitted_folder=args.path_videos_splitted,
    video_extensions=constants.VIDEO_EXTENSIONS,
    subclip_duration=constants.SUBCLIP_DURATION,
    shift_duration=constants.SHIFT_DURATION,
    starting_idx=args.starting_idx,
)

processor.move_arrived_videos()
processor.split_raw_videos()

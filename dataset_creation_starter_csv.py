"""
The following is for being used when creating the labeling csv file.
"""

from src.dataset_creation.videolabeler import VideoLabeler
from src import constants
import argparse

# Read with argparse the folder and the csv filename
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True,
                    help='Folder containing the videos to be labeled.')
parser.add_argument('--csv_filename', type=str, required=True,
                    help='Filename of the csv file to be created.')
args = parser.parse_args()

labeler = VideoLabeler(video_extensions=constants.VIDEO_EXTENSIONS)
labeler.create_starter_csv(
    folder=args.folder,
    csv_filename=args.csv_filename,
    )

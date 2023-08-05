"""
The following is for being used when the labeling is already done
and we want to move the files to the corresponding folders
"""

from src.dataset_creation.videolabeler import VideoLabeler
from src import constants
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source_folder', type=str, required=True,
                    help='Folder containing the input videos.')
parser.add_argument('--destination_folder', type=str, required=True,
                    help='Folder containing the output videos.')
parser.add_argument('--csv_filename', type=str, required=True,
                    help='Filename of the csv file to be created.')
args = parser.parse_args()

labeler = VideoLabeler(video_extensions=constants.VIDEO_EXTENSIONS)
labeler.read_dataframe(args.csv_filename)
labeler.move_files(
    source_folder=args.source_folder,
    destination_folder=args.destination_folder,
    )

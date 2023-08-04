from src.analysis import AnalyseDataset
from pathlib import Path
import argparse
from src import constants

# Read dataset path with argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to the dataset folder",
)
args = parser.parse_args()

analyser = AnalyseDataset(
    DATASET_FOLDER=args.dataset_path,
    VIDEO_EXTENSIONS=tuple(constants.VIDEO_EXTENSIONS),
    save_dir=Path("./report/dataset_analysis")
)

analyser.run()

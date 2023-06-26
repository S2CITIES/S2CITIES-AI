from src.analysis import analyse_dataset
from pathlib import Path

DATASET_FOLDER = Path("./data/4_videos_labeled/0")

analyse_dataset(DATASET_FOLDER)

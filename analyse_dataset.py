from src.analysis import AnalyseDataset
from pathlib import Path
import json

# Read from json file
with open("./src/const.json", "r", encoding="utf-8") as f:
    const = json.load(f)
    
analyser = AnalyseDataset(
    DATASET_FOLDER=Path("./data/4_videos_labeled"),
    VIDEO_EXTENSIONS=tuple(const["VIDEO_EXTENSIONS"]),
    save_dir=Path("./report/dataset_analysis")
)

analyser.run()

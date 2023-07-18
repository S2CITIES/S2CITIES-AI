import json
from pathlib import Path

from src.analysis import AnalyseTimeSeries

if __name__ == "__main__":

    # Read from json file
    with open("./src/const.json", "r", encoding="utf-8") as f:
        const = json.load(f)

    data_path = Path(const["DATA_PATH"])
    features_extracted_path = Path(const["FEATURES_EXTRACTED"])

    analyser = AnalyseTimeSeries(
        FOLDER=data_path/features_extracted_path
    )

    analyser.run()

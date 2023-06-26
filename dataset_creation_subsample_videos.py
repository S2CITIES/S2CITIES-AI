"""
The following is for being used when creating the dataset clips form the collected videos.
"""

from pathlib import Path
import os
import json

from src.dataset_creation.videosubsampler import VideoSubsampler

# Read from json file
with open("./src/const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

subsampler = VideoSubsampler(
    target_fps=const["SUBSAMPLE_FPS"],
    video_extensions=const["VIDEO_EXTENSIONS"],
    source_dir=Path(const["DATA_PATH"]) / const["VIDEOS_LABELED"],
    target_dir=Path(const["DATA_PATH"]) / const["VIDEOS_LABELED_SUBSAMPLED"]
    )

subsampler.subsample_videos()

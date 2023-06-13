"""
This file subsamples the videos in the dataset to a target fps.
"""

from pathlib import Path
import os
import json

from moviepy.video.io.VideoFileClip import VideoFileClip

# Set the target fps
TARGET_FPS = 12

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

# Read from json file
with open("const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

video_extensions = const["VIDEO_EXTENSIONS"]

# Define the input and output directories
input_dir = Path("dataset_creation/4_videos_labeled")
output_dir = Path("dataset_creation/4_videos_labeled_subsampled")

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

for target_dir in ["0", "1"]:

    # Create the target directory if it doesn't exist
    (output_dir / target_dir).mkdir(parents=True, exist_ok=True)

    # Loop through the files in the target directory
    for input_file in sorted((input_dir / target_dir).glob("*")):

        if input_file.is_file() and input_file.suffix in video_extensions:

            # Print the current file
            print(f"Processing {input_file}")

            output_file = output_dir / input_file.relative_to(input_dir)

            # Load the video clip
            clip = VideoFileClip(str(input_file))

            # Resample the video clip to the target fps
            clip_resampled = clip.set_fps(TARGET_FPS)

            # Write the resampled video clip to the output file
            if input_file.suffix.lower() == ".mov":
                clip_resampled.write_videofile(str(output_file), codec="libx264")
            else:
                clip_resampled.write_videofile(str(output_file))

            # Close the video clips
            clip.close()
            clip_resampled.close()

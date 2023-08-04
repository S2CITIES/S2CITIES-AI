"""
This file is used to extract the keypoints from the videos in the dataset by leveraging the KeypointsExtractor class.
"""

from pathlib import Path
import argparse

import numpy as np

from src.keypointsextractor import KeypointsExtractor
from src import constants

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Subsample videos.')
    parser.add_argument('--input', type=str, required=True, help='Path to input directory.')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory.')
    args = parser.parse_args()

    # Define paths
    path_input = Path(args.input)
    path_output = Path(args.output)

    # Create output folder if it doesn't exist
    path_output.mkdir(parents=True, exist_ok=True)

    # Loop through each label folder
    for target in ["0", "1"]:

        # Create the target directory if it doesn't exist
        (path_output / target).mkdir(parents=True, exist_ok=True)

        # Loop through each video in the input folder
        for input_file in sorted((path_input / target).glob("*")):

            # Skip if the input is not a file or if it is not a video
            if not (input_file.is_file() and input_file.suffix.lower() in constants.VIDEO_EXTENSIONS):
                continue
            
            # Define the output file
            output_file = (path_output / input_file.relative_to(path_input)).with_suffix(".npy")

            # If the keypoints file already exists, skip it
            if output_file.is_file():
                print(f"Skipping {input_file}")
                continue

            # Print the current file
            print(f"Processing {input_file}")
            
            # Extract keypoints from the video
            feature_extractor = KeypointsExtractor(str(input_file), show_image=True)
            keypoints = feature_extractor.extract_keypoints_from_video()

            # Save keypoints as npy file if keypoints is not None
            if keypoints is not None:
                np.save(output_file, keypoints)
            else:
                print("No hand detected for the whole video, or multiple hands detected.")

        # Loop through the files in the output directory
        for output_file in sorted((path_output / target).glob("*")):

            # If there is not a corresponding file in the input directory, delete it
            if not (path_input / output_file.relative_to(path_output)).is_file():
                print(f"Deleting {output_file}")
                output_file.unlink()

    print("Done extracting keypoints.")

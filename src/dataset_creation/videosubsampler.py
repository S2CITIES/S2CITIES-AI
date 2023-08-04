"""
This file subsamples the videos in the dataset to a target fps.
"""

from pathlib import Path
from typing import List
from moviepy.video.io.VideoFileClip import VideoFileClip

class VideoSubsampler:
    def __init__(self,
                 target_fps: int,
                 video_extensions: List[str],
                 path_input: str,
                 path_output: str):
        self.target_fps = target_fps
        self.video_extensions = video_extensions
        self.path_input = Path(path_input)
        self.path_output = Path(path_output)

    def subsample_videos(self):
        
        # Create the output directory if it doesn't exist
        self.path_output.mkdir(parents=True, exist_ok=True)

        # For every target
        for target in ["0", "1"]:

            # Create the target directory if it doesn't exist
            (self.path_output / target).mkdir(parents=True, exist_ok=True)

            # Loop through the files in the input directory
            for input_file in sorted((self.path_input / target).glob("*")):

                if not (input_file.is_file() and input_file.suffix.lower() in self.video_extensions):
                    continue

                output_file = self.path_output / input_file.relative_to(self.path_input)

                # If the output file already exists, skip it
                if output_file.is_file():
                    print(f"Skipping {input_file}")
                    continue

                # Print the current file
                print(f"Processing {input_file}")

                # Load the video clip
                clip = VideoFileClip(str(input_file))

                # Resample the video clip to the target fps
                clip_resampled = clip.set_fps(self.target_fps)

                # Define a dictionary of codecs
                codes_dict = {
                    ".mp4": "libx264",
                    ".mov": "libx264",
                    ".avi": "mpeg4",
                }
                # Get the codec from the dictionary, check if it exists
                codec = codes_dict.get(input_file.suffix.lower())
                # Write the resampled video clip to the output file
                clip_resampled.write_videofile(str(output_file), codec=codec)

                # Close the video clips
                clip.close()
                clip_resampled.close()

            # Loop through the files in the output directory
            for output_file in sorted((self.path_output / target).glob("*")):

                # If there is not a corresponding file in the input directory, delete it
                if not (self.path_input / output_file.relative_to(self.path_output)).is_file():
                    print(f"Deleting {output_file}")
                    output_file.unlink()
                
        print("Done subsampling videos.")
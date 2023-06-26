"""
This file subsamples the videos in the dataset to a target fps.
"""

from moviepy.video.io.VideoFileClip import VideoFileClip

class VideoSubsampler:
    def __init__(self, target_fps, video_extensions, source_dir, target_dir):
        self.target_fps = target_fps
        self.video_extensions = video_extensions
        self.source_dir = source_dir
        self.target_dir = target_dir

        # Create the output directory if it doesn't exist
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def subsample_videos(self):

        for target_dir in ["0", "1"]:

            # Create the target directory if it doesn't exist
            (self.target_dir / target_dir).mkdir(parents=True, exist_ok=True)

            # Loop through the files in the target directory
            for input_file in sorted((self.source_dir / target_dir).glob("*")):

                if input_file.is_file() and input_file.suffix.lower() in self.video_extensions:

                    output_file = self.target_dir / input_file.relative_to(self.source_dir)

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

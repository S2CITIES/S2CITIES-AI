""""
This module contains the VideoProcessor class, which is used to process the videos
in the dataset folder.
"""

from pathlib import Path
import os
import json
import cv2

from utils import get_video_files, move_file

class VideoProcessor:
    def __init__(self, const_file_path):
        # Read from json file
        with open(const_file_path, "r", encoding="utf-8") as f:
            const = json.load(f)

        # Define paths
        self.VIDEOS_ARRIVED = const["VIDEOS_ARRIVED"]
        self.VIDEOS_RAW = const["VIDEOS_RAW"]
        self.VIDEOS_RAW_PROCESSED = const["VIDEOS_RAW_PROCESSED"]
        self.VIDEOS_SPLITTED = const["VIDEOS_SPLITTED"]
        self.VIDEOS_LABELED = const["VIDEOS_LABELED"]

        # Create folders if necessary
        folders = [self.VIDEOS_SPLITTED, self.VIDEOS_ARRIVED, self.VIDEOS_RAW, self.VIDEOS_RAW_PROCESSED, self.VIDEOS_LABELED]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def move_arrived_videos(self):
        files = get_video_files(self.VIDEOS_ARRIVED)
        number = self.find_last_number(self.VIDEOS_RAW) + 1
        for file in files:
            destination_name = "vid_" + self.format_with_leading(number)
            destination_extension = Path(file).suffix
            destination_fullname = destination_name + destination_extension
            print(f"Moving {file} to {destination_fullname}")
            move_file(str(Path(self.VIDEOS_ARRIVED) / file), str(Path(self.VIDEOS_RAW) / destination_fullname))
            number += 1

    def split_raw_videos(self):
        files = get_video_files(self.VIDEOS_RAW)
        for file in files:
            self.cut_subclips(input_video=str(Path(self.VIDEOS_RAW) / file), output_folder=str(self.VIDEOS_SPLITTED), subclip_duration=3, shift_duration=2)
            move_file(source=str(Path(self.VIDEOS_RAW) / file), destination=str(Path(self.VIDEOS_RAW_PROCESSED) / file))
            print(f"Processed video {file}")

    def cleanup_folder(self, valid_extensions=[".mp4",".avi",".mov",".wmv",".flv"]):
        """
        This function goes through the files in the VIDEOS_ARRIVED folder
        and, if their extension is not in the ones in ext, it removes it.
        """
        files = get_video_files(self.VIDEOS_ARRIVED)
        for file in files:
            extension = Path(file).suffix.lower()
            if extension not in valid_extensions:
                (Path(self.VIDEOS_ARRIVED) / file).unlink()

    def cut_subclips(self, input_video: str, output_folder: str, subclip_duration=3, shift_duration=2) -> None:
        # Set video file path
        video_path = Path(input_video)

        # Create video capture object
        cap = cv2.VideoCapture(str(video_path))

        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get total number of frames in video
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate total number of subclips
        total_subclips = int((num_frames / fps - subclip_duration) / shift_duration) + 1

        # Set output folder name and create folder
        os.makedirs(output_folder, exist_ok=True)

        # Loop through each subclip and extract frames
        for i in range(total_subclips):
            # Calculate start and end frame indexes for current subclip
            start_frame = int(i * shift_duration * fps)
            end_frame = int(start_frame + subclip_duration * fps)

            # Set output file name and path for current subclip
            sub_id = self.format_with_leading(i+1)
            output_file = self.append_id(input_video, sub_id)
            output_path = os.path.join(output_folder, output_file)

            # Create video writer object for current subclip
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            # Loop through frames and write to subclip
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for j in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                # Flip frame vertically
                frame = cv2.flip(frame, 0)
                out.write(frame)

            # Release video writer object for current subclip
            out.release()

            # Print progress
            print(f"Processed subclip {i+1}/{total_subclips}")

        # Release video capture object
        cap.release()

    @staticmethod
    def find_last_number(folder_name: str) -> int:
        files = get_video_files(folder_name)
        if len(files) == 0:
            return 0
        else:
            last_filename = sorted(files)[-1]
            basename = last_filename.split("_")[1][:-4]
            return int(basename)

    @staticmethod
    def format_with_leading(number: int) -> str:
        return "{:05d}".format(number)

    @staticmethod
    def append_id(filename, id) -> str:
        p = Path(filename)
        return f"{p.stem}_{id}{p.suffix}"
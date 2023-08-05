import time
from pathlib import Path
import os
import cv2
import pandas as pd

from src.utils import get_video_files, move_file

class VideoLabeler:
    def __init__(self, video_extensions):
        
        self.VIDEO_EXTENSIONS = tuple(video_extensions)
        self.dataframe = None
    
    def read_dataframe(self, csv_filename):
        # Read from csv file
        self.dataframe = pd.read_csv(csv_filename)
        

    def label_videos(self, source_folder):

        # Iterate over the dataframe
        for index, video in self.dataframe.iterrows():
            
            # Skip if video is already processed
            # i.e. if the label has already been set to either 0 or 1
            if not video["label"] == -1:
                continue

            file_path = os.path.join(source_folder, video["file"])
            
            cap = cv2.VideoCapture(file_path)
            
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    time.sleep(1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                cv2.imshow('Video Player', frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('1'):
                    # Set label to 1
                    print(f"Labeling video {video['file']} as 1")
                    self.dataframe.at[index, "label"] = 1
                    break
                elif key == ord('0'):
                    # Set label to 0
                    print(f"Labeling video {video['file']} as 0")
                    self.dataframe.at[index, "label"] = 0
                    break
                elif key == ord('q'):
                    return

            cap.release()
            cv2.destroyAllWindows()

    def update_csv(self, csv_filename):
        self.dataframe.to_csv(csv_filename, index=False)

    def create_starter_csv(self, folder, csv_filename):
        video_files = get_video_files(folder, self.VIDEO_EXTENSIONS)
        video_info = [{"file": file, "label": -1} for file in video_files]
        df = pd.DataFrame(video_info)
        # Sort by file name
        df = df.sort_values(by=["file"])
        df.to_csv(csv_filename, index=False)
    
    # Read the csv and move files to the corresponding folder based on the label
    def move_files(self, source_folder, destination_folder):

        # Create destination folder if it doesn't exist
        destination_folder = Path(destination_folder)
        destination_folder.mkdir(parents=True, exist_ok=True)

        # Convert to Path object
        source_folder = Path(source_folder)

        # Define the destination folders
        VIDEOS_LABEL_0_FOLDER = destination_folder / "0"
        VIDEOS_LABEL_1_FOLDER = destination_folder / "1"
        
        # Create folders if they don't exist
        VIDEOS_LABEL_0_FOLDER.mkdir(parents=True, exist_ok=True)
        VIDEOS_LABEL_1_FOLDER.mkdir(parents=True, exist_ok=True)

        for index, video in self.dataframe.iterrows():
            if video["label"] == 1:
                move_file(str(source_folder / video['file']), str(VIDEOS_LABEL_1_FOLDER / video['file']))
            elif video["label"] == 0:
                move_file(str(source_folder / video['file']), str(VIDEOS_LABEL_0_FOLDER / video['file']))
            else:
                print("File {} has no label".format(video["file"]))

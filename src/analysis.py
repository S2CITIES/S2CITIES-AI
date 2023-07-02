import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import src.utils as utils

# Set resolution of the plots to retina
#plt.rcParams['figure.dpi'] = 300

class AnalyseDataset:
    def __init__(self, DATASET_FOLDER, VIDEO_EXTENSIONS, save_dir=None):
        self.DATASET_FOLDER = DATASET_FOLDER
        self.VIDEO_EXTENSIONS = VIDEO_EXTENSIONS
        # Check that the dataset folder exists and contains only two folders, one 1 and one 0
        if not self.DATASET_FOLDER.exists():
            raise ValueError("Dataset folder does not exist")
        if len(os.listdir(self.DATASET_FOLDER)) != 2:
            raise ValueError(f"Dataset folder must contain two folders, but contains {os.listdir(self.DATASET_FOLDER)}")
        if not os.path.isdir(self.DATASET_FOLDER / "0"):
            raise ValueError("Dataset folder must contain a folder named 0")
        if not os.path.isdir(self.DATASET_FOLDER / "1"):
            raise ValueError("Dataset folder must contain a folder named 1")
        
        # Get video files
        self.video_files0 = utils.get_video_files(str(DATASET_FOLDER / "0"), self.VIDEO_EXTENSIONS)
        self.video_files1 = utils.get_video_files(str(DATASET_FOLDER / "1"), self.VIDEO_EXTENSIONS)
        self.video_files = self.video_files0 + self.video_files1

        self.save_dir = save_dir
        # Create the save directory if it does not exist
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        # Create an empty list to store the video information
        video_info = []

        # Loop through each list of video files
        for video_files, target_class in zip([self.video_files0, self.video_files1], ["0", "1"]):

            # Loop through each video file
            for video_file in video_files:

                # Open the video file
                cap = cv2.VideoCapture(str(self.DATASET_FOLDER / target_class / video_file))

                # Get the frame rate of the video
                frame_rate = cap.get(cv2.CAP_PROP_FPS)

                # Get the height and width of the video
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Calculate the aspect ratio of the video
                aspect_ratio = width / height

                # Get the duration of the video in frames and seconds
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_frames = num_frames
                duration_seconds = num_frames / frame_rate

                # Add the video information to the list
                video_info.append({
                    "file": str(video_file),
                    "duration_frames": duration_frames,
                    "duration_seconds": duration_seconds,
                    "frame_rate": frame_rate,
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                    "target_class": target_class
                })

                # Release the video file
                cap.release()

        # Create a dataframe from the video information
        df = pd.DataFrame(video_info)

        # Print the number of videos in each class and in total
        print("Number of videos in class 0:", len(self.video_files0))
        print("Number of videos in class 1:", len(self.video_files1))
        print("Number of videos in total", len(self.video_files)) 
        
        # Barplot of the number of videos per class
        fig, ax = plt.subplots()
        df["target_class"].value_counts().plot(kind="bar")
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of videos")
        ax.set_title("Number of videos per class")
        plt.xticks(rotation=0)
        if self.save_dir is not None:
            plt.savefig(self.save_dir / 'num_videos_per_class.pdf')
        plt.show()

        results = df.pivot_table(
            index="target_class",
            values=["duration_seconds", "frame_rate", "aspect_ratio"],
            aggfunc="mean",
            margins=True,
            margins_name='Total'
            )
        
        print(results)

        if self.save_dir is not None:
            results.to_latex(self.save_dir / 'video_statistics.tex')

        # Add a column to the dataframe with the orientation of the video
        # If the aspect ratio is greater than 1, the video is landscape
        # If the aspect ratio is less than 1, the video is portrait
        df["orientation"] = df["aspect_ratio"].apply(lambda x: "landscape" if x > 1 else "portrait")

        # Barplot of the number of videos per orientation
        fig, ax = plt.subplots()
        #df["orientation"].value_counts().plot(kind="bar")
        sns.countplot(x="orientation", hue="target_class", data=df, ax=ax)
        ax.set_xlabel("Orientation")
        ax.set_ylabel("Number of videos")
        ax.set_title("Number of videos per orientation")
        plt.xticks(rotation=0)
        if self.save_dir is not None:
            plt.savefig(self.save_dir / 'num_videos_per_orientation.pdf')
        plt.show()

        # Sort by duration in seconds
        #df = df.sort_values(by="duration_seconds")


class AnalyseTimeSeries:
    def __init__(self, FOLDER):

        self.FOLDER = FOLDER

        self.X = []
        self.y = []

        # To be wrapped in a time serier loader

        for class_target in [0, 1]:
            for file in self.FOLDER.glob(f"{class_target}/*.npy"):
                self.X.append(np.load(file))
                self.y.append(class_target)

        self.X = np.array(self.X)
        self.y = np.array(self.y)
    
        self.class_labels = {
            0: 'SFH_No',
            1: 'SFH_Yes',
        }

        # Set numpy seed for reproducibility
        np.random.seed(42)

    def run(self):


        # Exploration
        print(self.X.shape)
        print(self.y.shape)


        # Plot timeseries for each class
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle('Example of timeseries for each class')

        axs = axs.flatten()

        for i in range(len(self.class_labels.keys())):
            indicies = np.where(self.y == i)[0]
            random_index = np.random.choice(indicies, size=1)[0]
            sns.lineplot(ax=axs[i], data = self.X[random_index])
            axs[i].set_title(self.class_labels[i])
            axs[i].get_legend().remove()
            axs[i].set_xticks([])
        plt.show()


        # We also notice that the classes are in order:
        plt.figure(figsize=(10,4))
        sns.lineplot(data = self.y)
        plt.title('y')
        plt.ylabel('Class')
        plt.xlabel('Element')
        plt.show()


        plt.figure(figsize=(15,6))
        counts = {self.class_labels[k]: np.count_nonzero(self.y == k)
                  for k in self.class_labels.keys()}
        sns.barplot(x=list(counts), y=list(counts.values()))
        plt.title('Elements per class')
        plt.show()
        print(counts)



        plt.figure(figsize=(15,6))
        sns.boxplot(data = [self.X[:, :, f].flatten() for f in range(self.X.shape[-1])])
        plt.title('Features')
        plt.show()

        print("Global mean:", np.around(np.mean(self.X[:, :]), 5))
        print("Global std:", np.around(np.std(self.X[:, :]), 5))
        print("Global median:", np.around(np.median(self.X[:, :]), 5))
        print("Global min:", np.around(np.min(self.X[:, :]), 5))
        print("Global max:", np.around(np.max(self.X[:, :]), 5))
        for f in range(self.X.shape[-1]):
            print(f"Feature {f}: mean={np.around(np.mean(self.X[:, :, f]), 5)}, std={np.around(np.std(self.X[:, :, f]), 5)}, median={np.around(np.median(self.X[:, :, f]), 5)}")


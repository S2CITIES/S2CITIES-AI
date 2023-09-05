import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import constants, utils


def main(
    DATASET_FOLDER,
    SAVE_DIR,
):
    DATASET_FOLDER = Path(DATASET_FOLDER)
    SAVE_DIR = Path(SAVE_DIR)
    # Create the save directory if it does not exist
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Check that the dataset folder exists and contains only two folders, one 1 and one 0
    if not DATASET_FOLDER.exists():
        raise ValueError("Dataset folder does not exist")
    if not os.path.isdir(DATASET_FOLDER / "0"):
        raise ValueError("Dataset folder must contain a folder named 0")
    if not os.path.isdir(DATASET_FOLDER / "1"):
        raise ValueError("Dataset folder must contain a folder named 1")

    # Get video files
    video_files0 = utils.get_video_files(
        str(DATASET_FOLDER / "0"), tuple(constants.VIDEO_EXTENSIONS)
    )
    video_files1 = utils.get_video_files(
        str(DATASET_FOLDER / "1"), tuple(constants.VIDEO_EXTENSIONS)
    )
    video_files = video_files0 + video_files1

    # Create an empty list to store the video information
    video_info = []

    # Loop through each list of video files
    for video_files, target_class in zip([video_files0, video_files1], ["0", "1"]):
        # Loop through each video file
        for video_file in video_files:
            # Open the video file
            cap = cv2.VideoCapture(str(DATASET_FOLDER / target_class / video_file))

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
            video_info.append(
                {
                    "file": str(video_file),
                    "duration_frames": duration_frames,
                    "duration_seconds": duration_seconds,
                    "frame_rate": frame_rate,
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                    "target_class": target_class,
                }
            )

            # Release the video file
            cap.release()

    # Create a dataframe from the video information
    df = pd.DataFrame(video_info)

    # Print the number of videos in each class and in total
    print("Number of videos in class 0:", len(video_files0))
    print("Number of videos in class 1:", len(video_files1))
    print("Number of videos in total", len(video_files))

    # Barplot of the number of videos per class
    fig, ax = plt.subplots()
    df["target_class"].value_counts().plot(kind="bar")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of videos")
    ax.set_title("Number of videos per class")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "num_videos_per_class.pdf")
    # plt.show()

    results = df.pivot_table(
        index="target_class",
        values=["duration_seconds", "frame_rate", "aspect_ratio"],
        aggfunc="mean",
        margins=True,
        margins_name="Total",
    )

    print(results)

    # There is a bug in the format method of the dataframe
    results.style.format(escape="latex").to_latex(SAVE_DIR / "video_statistics.tex")

    # Add a column to the dataframe with the orientation of the video
    # If the aspect ratio is greater than 1, the video is landscape
    # If the aspect ratio is less than 1, the video is portrait
    df["orientation"] = df["aspect_ratio"].apply(
        lambda x: "landscape" if x > 1 else "portrait"
    )

    # Barplot of the number of videos per orientation
    fig, ax = plt.subplots()
    # df["orientation"].value_counts().plot(kind="bar")
    sns.countplot(x="orientation", hue="target_class", data=df, ax=ax)
    ax.set_xlabel("Orientation")
    ax.set_ylabel("Number of videos")
    ax.set_title("Number of videos per orientation")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "num_videos_per_orientation.pdf")
    # plt.show()

    # Sort by duration in seconds
    # df = df.sort_values(by="duration_seconds")


if __name__ == "__main__":
    # Read dataset path with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset folder",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Folder to save the plots",
    )

    args = parser.parse_args()

    main(args.dataset_path, args.save_dir)

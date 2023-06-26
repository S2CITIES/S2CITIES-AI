from pathlib import Path
import os
import json
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Import utils.py
from utils import get_video_files

# Set resolution of the plots to retina
#plt.rcParams['figure.dpi'] = 300

def analyse_dataset(DATASET_FOLDER):

    # Read from json file
    with open("./src/const.json", "r", encoding="utf-8") as f:
        const = json.load(f)

    # Get video files
    video_files = get_video_files(DATASET_FOLDER, tuple(const["VIDEO_EXTENSIONS"]))

    # Create an empty list to store the video information
    video_info = []

    # Loop through each video file
    for video_file in video_files:

        # Open the video file
        cap = cv2.VideoCapture(str(DATASET_FOLDER / video_file))

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
            "aspect_ratio": aspect_ratio
        })

        # Release the video file
        cap.release()

    # Create a dataframe from the video information
    df = pd.DataFrame(video_info)

    # Print average duration seconds
    print("Mean duration in seconds:", df["duration_seconds"].mean())

    # Print average frame rate
    print("Mean framerate in seconds:", df["frame_rate"].mean())

    # Print frame rate histogram
    fig, ax = plt.subplots()
    df["frame_rate"].hist()
    ax.set_xlabel("Frame rate")
    ax.set_ylabel("Number of videos")
    ax.set_title("Frame rate histogram")
    #plt.savefig('frame_rate.png')
    plt.show()

    # Add a column to the dataframe with the orientation of the video
    # If the aspect ratio is greater than 1, the video is landscape
    # If the aspect ratio is less than 1, the video is portrait
    df["orientation"] = df["aspect_ratio"].apply(lambda x: "landscape" if x > 1 else "portrait")

    # Barplot of the number of videos per orientation
    fig, ax = plt.subplots()
    df["orientation"].value_counts().plot(kind="bar")
    ax.set_xlabel("Orientation")
    ax.set_ylabel("Number of videos")
    ax.set_title("Number of videos per orientation")
    #plt.savefig('orientation.png')
    plt.show()

    # Sort by duration in seconds
    df = df.sort_values(by="duration_seconds")
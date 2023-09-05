"""
Utility functions for file management.
"""

import os
from pathlib import Path


def get_video_files(folder_path, video_extensions):
    """
    Returns a list of all files with the specified video extensions.
    """

    folder_path = Path(folder_path)
    video_files = [
        file.name
        for file in folder_path.iterdir()
        if file.is_file() and file.name.lower().endswith(video_extensions)
    ]

    # Sort the list of video files alphabetically
    video_files.sort()

    return video_files


def move_file(source, destination):
    """
    Moves a file from source to destination.
    """
    os.rename(source, destination)

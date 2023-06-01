from pathlib import Path
import os
from os import listdir
from os.path import isfile, join
import cv2
import json

from utils import get_video_files

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

# Read from json file
with open("../const.json", "r", encoding="utf-8") as f:
    const = json.load(f)

# define paths
VIDEOS_ARRIVED = const["VIDEOS_ARRIVED"]
VIDEOS_RAW = const["VIDEOS_RAW"]
VIDEOS_RAW_PROCESSED = const["VIDEOS_RAW_PROCESSED"]
VIDEOS_SPLITTED = const["VIDEOS_SPLITTED"]
VIDEOS_LABELED = const["VIDEOS_LABELED"]

# create them if necessary
folders = [VIDEOS_SPLITTED, VIDEOS_ARRIVED, VIDEOS_RAW, VIDEOS_RAW_PROCESSED, VIDEOS_LABELED]
for folder in folders:
    os.makedirs(folder, exist_ok=True)


# wrap this in a function
def cut_subclips(input_video: str, output_folder: str, subclip_duration=3, shift_duration=2) -> None:

    # Set video file path
    video_path = Path() / input_video

    # Create video capture object
    with cv2.VideoCapture(str(video_path)) as cap:

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
            sub_id = format_with_leading(i+1)
            output_file = append_id(input_video, sub_id)

            #output_file = f"subclip_{i}.mp4"
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
                out.write(frame)
            
            # Release video writer object for current subclip
            out.release()


def count_files(path):
    return len(get_file_list(path))

def get_file_list(path:str):
    """
    Returns a list of strings of all the names in the path, excluding the macOS
    system file ".DS_Store".
    """
    return [f for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store"]

def find_last_number(folder_name:str):
    onlyfiles = get_file_list(folder_name)
    if len(onlyfiles) == 0:
        return 0
    else:
        last_filename = sorted(onlyfiles)[-1]
        basename = last_filename.split("_")[1][:-4]
        return int(basename)

def move_file(source: str, destination: str) -> None:
    os.rename(source, destination)


def format_with_leading(number: int) -> str:
    return "{:05d}".format(number)

def move_arrived_videos():
    files = get_file_list(VIDEOS_ARRIVED)
    number = find_last_number(VIDEOS_RAW_PROCESSED)+1 # check that VIDEOS_RAW_PROCESSED is correct, the last may be on that or VIDEOS_RAW...
    for file in files:
        destination_name = "vid_"+format_with_leading(number)
        destination_extension = Path(file).suffix # it includes the "."
        destination_fullname = destination_name + destination_extension
        move_file(str(Path(VIDEOS_ARRIVED) / file), str(Path(VIDEOS_RAW) / destination_fullname))
        number = number + 1

def append_id(filename, id) -> str:
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, id)

def split_arrived_videos():
    files = get_file_list(VIDEOS_RAW)
    for file in files:
        cut_subclips(input_video=Path(VIDEOS_RAW) / file, output_folder=VIDEOS_SPLITTED, subclip_duration=3, shift_duration=2)
        move_file(source=Path(VIDEOS_RAW) / file, destination=Path(VIDEOS_RAW_PROCESSED) / file)


def cleanup_folder(valid_extensions=[".mp4",".avi",".mov",".wmv",".flv"]):
    """
    This function goes throught the files in the VIDEOS_ARRIVED folder
    and, if their extension is not in the ones in ext, it removes it.
    """
    files = get_file_list(VIDEOS_ARRIVED)
    for file in files:
        extension = Path(file).suffix.lower()
        if extension not in valid_extensions:
            (Path(VIDEOS_ARRIVED) / file).unlink()


#cleanup_folder()
#move_arrived_videos()
#split_arrived_videos() # TODO forse c'è un problema per cui il video viene sottosopra

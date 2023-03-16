from pathlib import Path
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import cv2

# define paths
VIDEOS_ARRIVED = "0_videos_arrived"
VIDEOS_RAW = "1_videos_raw"
VIDEOS_RAW_PROCESSED = "2_videos_raw_processed"
VIDEOS_SPLITTED = "3_videos_splitted"

# create them if necessary
folders = [VIDEOS_SPLITTED, VIDEOS_ARRIVED, VIDEOS_RAW, VIDEOS_RAW_PROCESSED]
for folder in folders:
    os.makedirs(folder, exist_ok=True)


# wrap this in a function
def cut_subclips(input_video: str, output_folder: str, subclip_duration=3, shift_duration=2, fps=24) -> None:

    # Set video file path
    video_path = Path() / input_video

    # Create video capture object
    with cv2.VideoCapture(str(video_path)) as cap:

        #video_fps = cap.get(cv2.CAP_PROP_FPS)

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
    """
    This function counts the number of files in a given directory.

    Args:
        path (str or Path): The path to the directory to count files in.

    Returns:
        The number of files in the directory.

    Example Usage:
        count_files("/path/to/directory")
        # Returns: 42

    Explanation:
        This function takes a path to a directory as input, counts the number of files in the directory, and returns the count. The input path can be either a string or a Path object. The function only counts the number of files in the directory, and does not perform any filtering or recursion.
    """
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
    """
    This function moves a file from a source location to a destination location by renaming it.

    Args:
        source (str or Path): The path to the file to be moved.
        destination (str or Path): The path to the location where the file should be moved.

    Returns:
        None

    Raises:
        OSError: If the file could not be moved for some reason.

    Example Usage:
        move_file("source/filename.txt", "destination/new_filename.txt")

    Explanation:
        This function moves a file from the source location to the destination location by renaming it. The source and destination arguments can be either a string or a Path object. If the file cannot be moved, an OSError is raised.
    """
    os.rename(source, destination)


def format_with_leading(number: int) -> str:
    """
    This function formats an integer with leading zeros to a string of length 5.

    Args:
        number (int): The integer to format.

    Returns:
        str: The formatted string.

    Example Usage:
        format_with_leading(9)
        # Output: "00009"

    Explanation:
        This function takes an integer and formats it as a string with leading zeros to ensure that the string is of length 5. For example, if the input integer is 9, the output will be "00009".
    """
    return "{:05d}".format(number)

def move_arrived_videos():
    """
    This function moves video files from the VIDEOS_ARRIVED directory to the VIDEOS_RAW directory, renaming them in the process. 

    Args:
        None.

    Returns:
        None.

    Explanation:
        This function gets a list of all video files in the VIDEOS_ARRIVED directory, and renames each file with a new name based on the next number in the sequence. The new name has a prefix "vid_" followed by a zero-padded number. For example, the first file would be renamed as "vid_001.mp4".

        Then, each file is moved from VIDEOS_ARRIVED to VIDEOS_RAW directory, with the new name. If a file with the same name already exists in the VIDEOS_RAW directory, it will be overwritten.

        If VIDEOS_ARRIVED or VIDEOS_RAW directories do not exist, a FileNotFoundError is raised.

    """
    files = get_file_list(VIDEOS_ARRIVED)
    number = find_last_number(VIDEOS_RAW_PROCESSED)+1 # check that VIDEOS_RAW_PROCESSED is correct, the last may be on that or VIDEOS_RAW...
    for file in files:
        destination_name = "vid_"+format_with_leading(number)
        destination_extension = Path(file).suffix # it includes the "."
        destination_fullname = destination_name + destination_extension
        move_file(Path(VIDEOS_ARRIVED) / file, Path(VIDEOS_RAW) / destination_fullname)
        number = number + 1

def append_id(filename, id) -> str:
    """
    This function appends an ID to a filename, preserving the original extension.

    Args:
        filename (str or Path): The original filename.
        id (str or int): The ID to append to the filename.

    Returns:
        A new string containing the original filename with the ID appended and the extension preserved.

    Example Usage:
        append_id("file.txt", 123)
        # Returns: "file_123.txt"

    Explanation:
        This function takes an input filename and an ID, and returns a new string that is the original filename with the ID appended and the extension preserved. The input filename can be either a string or a Path object. The ID can be either a string or an integer. The function does not modify the original file, it only returns a new string with the modified filename.
    """
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, id)

def split_arrived_videos():
    """
    This function splits the videos in the VIDEOS_RAW folder into subclips, and moves the original videos to
    the VIDEOS_RAW_PROCESSED folder.

    Args:
        None.

    Returns:
        None.

    Example Usage:
        split_arrived_videos()

    Explanation:
        This function splits the videos in the VIDEOS_RAW folder into subclips with a duration of 3 seconds and a
        shift of 2 seconds, and saves the subclips in the VIDEOS_SPLITTED folder. The original videos are then moved
        to the VIDEOS_RAW_PROCESSED folder. The function does not return anything.
    """
    files = get_file_list(VIDEOS_RAW)
    for file in files:
        cut_subclips(input_video=Path(VIDEOS_RAW) file,
                     output_folder=VIDEOS_SPLITTED, subclip_duration=3, shift_duration=2, fps=24)
        move_file(source=Path(VIDEOS_RAW) / file, destination=Path(VIDEOS_RAW_PROCESSED) / file)


def cleanup_folder(ext=[".mp4",".avi",".mov",".wmv",".flv"]):
    """
    This function goes throught the files in the VIDEOS_ARRIVED folder
    and, if their extension is not in the ones in ext, it removes it.
    """
    files = get_file_list(VIDEOS_ARRIVED)
    for file in files:
        extension = Path(file).suffix.lower() # it includes the "."
        if extension not in ext:
            (Path(VIDEOS_ARRIVED) / file).unlink()


cleanup_folder()
move_arrived_videos()
# TODO forse c'è un problema per cui il video viene sottosopra
split_arrived_videos()





# dataframe stuff, ma ci serve? credo di no
FILE_NAME = 'processed.csv'
df = pd.DataFrame(columns=['video_name'])

df.head()
df.to_csv(FILE_NAME, index=False)

df2 = pd.read_csv(FILE_NAME)


for f in folders:
    print(f'Folder {f} contains {count_files(f)} files')

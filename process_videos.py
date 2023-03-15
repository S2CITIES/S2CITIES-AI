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
    # quanti file ci sono nella cartella
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

def move_file(source, destination):
    os.rename(source, destination)

def format_with_leading(number):
    return "{:05d}".format(number)

def move_arrived_videos():
    # prendi i video da 0_videos_arrived e rinominali giusti mettendoli in 1
    files = get_file_list(VIDEOS_ARRIVED)
    number = find_last_number(VIDEOS_RAW)+1
    for f in files:
        destination_name = "vid_"+format_with_leading(number)
        destination_extension = Path(f).suffix # it includes the "."
        destination_fullname = destination_name + destination_extension
        move_file(Path(VIDEOS_ARRIVED) / f, Path(VIDEOS_RAW) / destination_fullname)
        number = number + 1

def append_id(filename, id):
    p = Path(filename)
    return "{0}_{2}{1}".format(p.stem, p.suffix, id)

def split_arrived_videos():
    # prendi i video in 1 e fai il taglio, mettendoli in 2 gli originali, e gli split in 3
    files = get_file_list(VIDEOS_RAW)
    for f in files:
        # split the video and rename it with sub-index
        cut_subclips(input_video = Path(VIDEOS_RAW) / f,
                    output_folder = VIDEOS_SPLITTED, subclip_duration=3, shift_duration=2, fps=24)

        move_file(source=Path(VIDEOS_RAW) / f,destination=Path(VIDEOS_RAW_PROCESSED) / f)



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

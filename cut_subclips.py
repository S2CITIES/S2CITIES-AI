import cv2
import os

# Set video file name and path
video_file = "example.mp4"
video_path = os.path.join(os.getcwd(), video_file)

# Set frame rate (fps) for output video
fps = 24

# Set duration of subclips and time window shift
subclip_duration = 3 # seconds
shift_duration = 2 # seconds

# Create video capture object
cap = cv2.VideoCapture(video_path)

# Get total number of frames in video
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate total number of subclips
total_subclips = int((num_frames / fps - subclip_duration) / shift_duration) + 1

# Set output folder name and create folder
output_folder = "subclips"
os.makedirs(output_folder, exist_ok=True)

# Loop through each subclip and extract frames
for i in range(total_subclips):
    # Calculate start and end frame indexes for current subclip
    start_frame = int(i * shift_duration * fps)
    end_frame = int(start_frame + subclip_duration * fps)
    
    # Set output file name and path for current subclip
    output_file = f"subclip_{i}.mp4"
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

# Release video capture object
cap.release()

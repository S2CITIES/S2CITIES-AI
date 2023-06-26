import cv2
import os

desired_video_duration = 2.5

video_directory = "./Jester"
output_directory = "./Jester_videos"

print(os.listdir(video_directory))

for video_folder in os.listdir(video_directory):

    video_dir = os.path.join(video_directory, video_folder)
    output_file = os.path.join(output_directory, video_folder + '.mp4')

    file_list = os.listdir(video_dir)
    file_list.sort()

    fps = len(file_list) / desired_video_duration

    frame_path = os.path.join(video_dir, file_list[0])
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for filename in file_list:
        frame_path = os.path.join(video_dir, filename)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()


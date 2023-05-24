import shutil
import os
import cv2
import time

def move_video(file_path, destination_folder):
    filename = os.path.basename(file_path)
    destination_path = os.path.join(destination_folder, filename)
    shutil.move(file_path, destination_path)

def process_videos(source_folder, folder_0, folder_1):
    video_files = [file for file in os.listdir(source_folder) if file.endswith('.mp4')]

    for video_file in video_files:
        file_path = os.path.join(source_folder, video_file)
        print(file_path)
        cap = cv2.VideoCapture(file_path)
        
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print('no video')
                time.sleep(1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            cv2.imshow('Video Player', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('1'):
                move_video(file_path, folder_1)
                break
            elif key == ord('0'):
                move_video(file_path, folder_0)
                break
            elif key == ord('q'):
                return

        cap.release()
        cv2.destroyAllWindows()

# Set up your source folder and destination folders
source_folder = 'vid'
folder_0 = '0'
folder_1 = '1'

# Call the function to process the videos
process_videos(source_folder, folder_0, folder_1)

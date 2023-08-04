import cv2
import os

def convert_ratio(target_ratio, source_video_path, dest_video_path):

    for label in os.listdir(source_video_path):

        label_path = os.path.join(source_video_path, label)

        for video_file in os.listdir(label_path):

            current_video_path = os.path.join(label_path, video_file)

            input_video = cv2.VideoCapture(current_video_path)
            width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            aspect_ratio = width / height

            if aspect_ratio > target_ratio:
                # Original video is wider, crop the sides
                target_width = int(height * target_ratio)
                crop_left = (width - target_width) // 2
                crop_right = width - target_width - crop_left
                crop_top, crop_bottom = 0, 0
            else:
                # Original video is taller, crop the top and bottom
                target_height = int(width / target_ratio)
                crop_top = (height - target_height) // 2
                crop_bottom = height - target_height - crop_top
                crop_left, crop_right = 0, 0

            # Create an output video writer using OpenCV's VideoWriter class
            current_video_path_dest = os.path.join(dest_video_path, label, video_file)

            output_video = cv2.VideoWriter(current_video_path_dest,
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        input_video.get(cv2.CAP_PROP_FPS),
                                        (width - crop_left - crop_right, height - crop_top - crop_bottom))

            # Process each frame of the input video, apply cropping, and write to the output video
            while True:
                ret, frame = input_video.read()
                if not ret:
                    break

                cropped_frame = frame[crop_top:height - crop_bottom, crop_left:width - crop_right]
                output_video.write(cropped_frame)

            input_video.release()
            output_video.release()

def resize_frames(target_width, target_height, source_video_path, dest_video_path):

    for label in os.listdir(source_video_path):

        label_path = os.path.join(source_video_path, label)

        for video_file in os.listdir(label_path):

            current_video_path = os.path.join(label_path, video_file)

            input_video = cv2.VideoCapture(current_video_path)

            # Create an output video writer using OpenCV's VideoWriter class
            current_video_path_dest = os.path.join(dest_video_path, label, video_file)

            output_video = cv2.VideoWriter(current_video_path_dest,
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        input_video.get(cv2.CAP_PROP_FPS),
                                        (target_width, target_height))

            # Process each frame of the input video, apply cropping, and write to the output video
            while True:
                ret, frame = input_video.read()
                if not ret:
                    break

                new_frame = cv2.resize(frame, (target_height, target_width))
                output_video.write(new_frame)

            input_video.release()
            output_video.release()

if __name__ == '__main__':

    source_video_path = './dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives'
    dest_video_path = './dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1'
    
    convert_ratio(target_ratio=1, 
                  source_video_path=source_video_path,
                  dest_video_path=dest_video_path)

    source_video_path = './dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1'
    dest_video_path = './dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_test_new_negatives_ratio1_224x224'

    resize_frames(target_width=224, 
                  target_height=224, 
                  source_video_path=source_video_path,
                  dest_video_path=dest_video_path)
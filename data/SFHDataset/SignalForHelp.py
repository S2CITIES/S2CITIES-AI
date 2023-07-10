import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import transforms.spatial_transforms as SPtransforms
import transforms.temporal_transforms as TPtransforms
import os
import cv2
import mediapipe as mp
from PIL import Image

def load_video(video_path, temporal_transform=None, spatial_transform=None, sample_duration=16, norm_value=1.0, save_output=False):

        cap = cv2.VideoCapture(video_path)

        clip = []
        # TODO: Make it faster by only loading the sampled frames and save the total number of frames in the annotation file
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to a PIL Image
            frame = Image.fromarray(frame)
            # print(frame.size)
            clip.append(frame)

        cap.release()

        # Apply Temporal Transform
        if temporal_transform is not None:
            n_frames = len(clip)
            # print(f"Clip lenght: {n_frames}")
            frame_indices = list(range(1, n_frames+1))
            frame_indices = temporal_transform(frame_indices)
            # print(frame_indices)
            clip = [clip[i-1] for i in frame_indices]

        if spatial_transform is not None:
            # Apply Spatial Transform
            spatial_transform.randomize_parameters()
            clip = [spatial_transform(frame) for frame in clip]

        if save_output: # To "visualize" the effect of temporal and spatial transforms
            codec = cv2.VideoWriter_fourcc(*"mp4v")  # Video codec (e.g., "mp4v", "XVID")
            output_file = os.path.join('test', video_path.split('/')[-1])  # Output video file name
            print(output_file)
            frame_size = (112, 112)  # Frame size (width, height)
            fps = sample_duration/2.5

            video_writer = cv2.VideoWriter(output_file, codec, fps, frame_size)

            for frame in clip:
                np_img = frame.numpy()
                np_img = np.transpose(np_img, (1, 2, 0))
                np_img = np_img * norm_value
                np_img = np_img.astype(np.uint8)
                video_writer.write(np_img)

            video_writer.release()

        clip = torch.stack(clip, dim=0) # Tensor with shape TCHW
        clip = clip.permute(1, 0, 2, 3) # Tensor with shape CTHW

        return clip

class Signal4HelpDataset(Dataset):
    def __init__(self, annotation_path, temporal_transform, spatial_transform):
        
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.videos = []
        
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            fields = line.strip().split(' ')
            video_path = ' '.join([fields[:-1]])
            label = int(fields[-1])
            self.videos.append((video_path, label))

    # def extract_hand_bb(self, frame, frame_width, frame_height, first_only=True):
    #     # Apparently the process() function is not thread-safe. With multiple workers, the code gets stuck here.
    #     results = self.hands_model.process(frame)
    #     #print("Hands Detector Model is ending processing for this frame.")
    #     ROIs = []

    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             x_min = y_min = float("inf")
    #             x_max = y_max = float("-inf")
    #             for landmark in hand_landmarks.landmark:
    #                 x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
    #                 # print(f"Landmarks: {x, y}")
    #                 if x < x_min:
    #                     x_min = x
    #                 if x > x_max:
    #                     x_max = x
    #                 if y < y_min:
    #                     y_min = y
    #                 if y > y_max:
    #                     y_max = y

    #             # print(y_min, y_max, x_min, x_max)
    #             # Define the desired aspect ratio
    #             aspect_ratio = 1  # Example: 16:9 aspect ratio

    #             # Calculate the center coordinates of the hand landmarks
    #             x_center = (x_min + x_max) // 2
    #             y_center = (y_min + y_max) // 2

    #             # print(f"Frame dims: {frame_width, frame_height}")
    #             # print(f"Center: {x_center, y_center}")

    #             # Calculate the width and height of the bounding box
    #             width = max(x_max - x_min, (y_max - y_min) * aspect_ratio)
    #             height = max(y_max - y_min, (x_max - x_min) / aspect_ratio)

    #             # Calculate the new bounding box coordinates based on the center, width, and height
    #             x_min = int(x_center - width / 2)
    #             x_max = int(x_center + width / 2)
    #             y_min = int(y_center - height / 2)
    #             y_max = int(y_center + height / 2)

    #             # Check if one between x_min or y_min becomes negative
    #             if x_min < 0:
    #                 x_min = 0
    #             if y_min < 0:
    #                 y_min = 0
                
    #             # Check if one between x_max or y_max is higher than frame_width or frame_height
    #             if x_max > frame_width:
    #                 x_max = frame_width
    #             if y_max > frame_height:
    #                 y_max = frame_height

    #             # Crop the ROI box from the frame
    #             # print(frame.shape)
    #             roi = frame[y_min:y_max, x_min:x_max]
    #             # print(y_min, y_max, x_min, x_max)
    #             # print(f"Roi shape before reshaping: {roi.shape}")
    #             # print(roi.shape)

    #             # Padding to reach the desired resolution
    #             pad_width = ((0, int(height - roi.shape[0])), (0, int(width - roi.shape[1])), (0,0))
    #             # print(pad_width)
    #             roi = np.pad(roi, pad_width, mode='constant')
    #             roi = cv2.resize(roi, (self.image_height, self.image_width))
    #             # print(f"Roi shape after reshaping: {roi.shape}")
    #             # print(roi.shape)
    #             ROIs.append(roi)
    #     else:
    #         return None
        
    #     if first_only:
    #         return ROIs[0]
    #     else:
    #         return ROIs

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_path, label = self.videos[index]
        video = load_video(video_path, 
                           temporal_transform=self.temporal_transform,
                           spatial_transform=self.spatial_transform,
                           save_output=False)
        return video, label

if __name__ == '__main__':

    spatial_transform = SPtransforms.Compose([
        SPtransforms.Scale(112),
        SPtransforms.CenterCrop(112),
        SPtransforms.ToTensor(1.0),
        SPtransforms.Normalize(
                        mean=[
                            124.02363586425781,
                            114.20242309570312,
                            103.32056427001953
                        ], 
                        std=[
                            61.589691162109375,
                            61.51222610473633,
                            60.233455657958984
                        ])
    ])

    temporal_transform = TPtransforms.TemporalRandomCrop(16, 1)

    # Test load_video 
    load_video(video_path='../../dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_simplified_ratio1/1/vid_00053_00006.mp4',
               temporal_transform=temporal_transform,
               spatial_transform=spatial_transform,
               norm_value=1.0,
               save_output=True)
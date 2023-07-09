import torch
import random
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import os
import sys
from tqdm import tqdm
import pickle
import cv2
import mediapipe as mp
from pytorchvideo.transforms import UniformTemporalSubsample
import matplotlib.pyplot as plt

class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample

        rand_end = max(0, vid_duration - clip_duration - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + clip_duration, vid_duration)

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return selected_frames

class Signal4HelpDataset(Dataset):
    def __init__(self, annotation_path, image_width, image_height, temporal_transform, spatial_transform):
        
        self.video_path = video_path
        self.image_width = image_width
        self.image_height = image_height
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform

        self.videos = []
        
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            video_path, label = line.strip().split(' ')
            self.videos.append((video_path, int(label)))

    def load_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        # print(cap)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (self.image_height, self.image_width))
            frames.append(frame_rgb)

        cap.release()

        # TODO: Apply spatial and temporal transforms here
        video = torch.stack([transforms.ToTensor()(frame) for frame in frames])
        video = UniformTemporalSubsample(num_samples=16, temporal_dim=0)(video)

        # selected_frames = []

        # for i in range(video.shape[0]):
        #     selected_frames.append(video[i])

        # codec = cv2.VideoWriter_fourcc(*"mp4v")  # Video codec (e.g., "mp4v", "XVID")
        # output_file = os.path.join('data/SFHDataset/test', load_video_path.split('/')[-1])  # Output video file name
        # frame_size = (224, 224)  # Frame size (width, height)
        # fps = 16/2.5

        # video_writer = cv2.VideoWriter(output_file, codec, fps, frame_size)

        # for frame in selected_frames:
        #     np_img = frame.numpy()
        #     np_img = np.transpose(np_img, (1, 2, 0))
        #     np_img = np_img * 255.0
        #     np_img = np_img.astype(np.uint8)
        #     np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        #     #plt.figure()
        #     #plt.imshow(np_img) 
        #     #plt.show()  # display it
        #     # Assuming the frame is in NumPy array format
        #     video_writer.write(np_img)

        # print(f"Saving {output_file}")
        # video_writer.release()


        return video

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
        video = self.load_video(video_path)
        return video, label

if __name__ == '__main__':
    video_path = "./SFH/SFH_Dataset_S2CITIES_ratio1_224x224"
    dataset_name = "./"

    # Create the VideoDataset and DataLoader
    dataset = Signal4HelpDataset(video_path, 
                                 image_width=224, 
                                 image_height=224)
    
    # Check that the dataset has correctly been created
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    for idx, batch in enumerate(dataloader):
        # print(batch) 
        break


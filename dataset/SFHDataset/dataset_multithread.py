import torch
import numpy as np
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
import os
import pickle
import cv2
import mediapipe as mp
from pytorchvideo.transforms import UniformTemporalSubsample
import multiprocessing

# NOTE: To be fixed. Currently, it doesn't work because of Pickling Errors due to multiprocessing + mediapipe

class Signal4HelpDataset(Dataset):
    def __init__(self, video_path, image_width, image_height, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.image_width = image_width
        self.image_height = image_height
        self.hands_model = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.videos = self.load_videos()

    def process_videos(self, paths):
        print("Starting a process")
        videos = [] # Local memory to each process
        for video_path, label in paths:

            cap = cv2.VideoCapture(video_path)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            regions = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hand_region = self.extract_hand_bb(frame, frame_width, frame_height, first_only=True)
                if hand_region is None:
                    hand_region = np.zeros((self.image_height, self.image_width, 3))
                regions.append(hand_region)

            cap.release()
            video = torch.stack([transforms.ToTensor()(region) for region in regions])
            if self.transform:
                video = self.transform(video)

            videos.append((video, label))
            print("Ending processing of video")

        return videos
    
    def load_videos(self):
        
        path_list = []
        # First Step: Build a list of video paths to process
        for label in os.listdir(self.video_path):
            label_path = os.path.join(self.video_path, label)
            for video_file in os.listdir(label_path):
                video_path = os.path.join(label_path, video_file)
                path_list.append((video_path, int(label)))

        num_processes = 8
        chunk_size = len(path_list) // num_processes
        chunks = [path_list[i:i+chunk_size] for i in range(0, len(path_list), chunk_size)]

        pool = multiprocessing.Pool(processes=num_processes)
        processed_chunks = pool.map(self.process_videos, chunks)

        processed_videos = [video for chunk in processed_chunks for video in chunk]

        return processed_videos

    def extract_hand_bb(self, frame, frame_width, frame_height, first_only=True):
        results = self.hands_model.process(frame)
        ROIs = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = y_min = float("inf")
                x_max = y_max = float("-inf")
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    # print(f"Landmarks: {x, y}")
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y

                # print(y_min, y_max, x_min, x_max)
                # Define the desired aspect ratio
                aspect_ratio = 1  # Example: 16:9 aspect ratio

                # Calculate the center coordinates of the hand landmarks
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2

                # print(f"Frame dims: {frame_width, frame_height}")
                # print(f"Center: {x_center, y_center}")

                # Calculate the width and height of the bounding box
                width = max(x_max - x_min, (y_max - y_min) * aspect_ratio)
                height = max(y_max - y_min, (x_max - x_min) / aspect_ratio)

                # Calculate the new bounding box coordinates based on the center, width, and height
                x_min = int(x_center - width / 2)
                x_max = int(x_center + width / 2)
                y_min = int(y_center - height / 2)
                y_max = int(y_center + height / 2)

                # Check if one between x_min or y_min becomes negative
                if x_min < 0:
                    x_min = 0
                if y_min < 0:
                    y_min = 0
                
                # Check if one between x_max or y_max is higher than frame_width or frame_height
                if x_max > frame_width:
                    x_max = frame_width
                if y_max > frame_height:
                    y_max = frame_height

                # Crop the ROI box from the frame
                # print(frame.shape)
                roi = frame[y_min:y_max, x_min:x_max]
                # print(y_min, y_max, x_min, x_max)
                # print(f"Roi shape before reshaping: {roi.shape}")
                # print(roi.shape)

                # Padding to reach the desired resolution
                pad_width = ((0, int(height - roi.shape[0])), (0, int(width - roi.shape[1])), (0,0))
                # print(pad_width)
                roi = np.pad(roi, pad_width, mode='constant')
                roi = cv2.resize(roi, (self.image_height, self.image_width))
                # print(f"Roi shape after reshaping: {roi.shape}")
                # print(roi.shape)
                ROIs.append(roi)
        else:
            return None
        
        if first_only:
            return ROIs[0]
        else:
            return ROIs

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video, label = self.videos[index]
        return video, label

# Generate dataset (extracting hand bounding boxes from videos) and save preprocessed data
# This avoid to repeat a lot of expensive work each time a batch is loaded for training
def save_preprocessed_data(video_path, dest_path):
    # Set the video path
    video_path = "./SFH/SFH_Dataset_S2CITIES"

    # Define any transforms you want to apply to the videos
    transform = transforms.Compose([
        UniformTemporalSubsample(num_samples=16, temporal_dim=0),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create the VideoDataset and DataLoader
    dataset = Signal4HelpDataset(video_path, image_width=112, image_height=112, transform=transform)

    # Make a Train-Test Split
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_path = os.path.join(dest_path, 'train_set.pkl')
    test_path = os.path.join(dest_path, 'test_set.pkl')
    with open(train_path, "wb") as file:
        pickle.dump(train_dataset, file)
    with open(test_path, "wb") as file:
        pickle.dump(test_dataset, file)

if __name__ == '__main__':
    # Set the video path
    video_path = "./SFH/SFH_Dataset_S2CITIES"
    dest_path = "./SFH/preprocessed_data"
    save_preprocessed_data(video_path, dest_path)


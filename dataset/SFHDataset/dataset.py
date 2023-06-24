import torch
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

class Signal4HelpDataset(Dataset):
    def __init__(self, video_path, image_width, image_height,
                 dataset_source, 
                 extract_bb_region=True, preprocessing_on=True, load_on_demand=False, 
                 transform=None):
        
        self.video_path = video_path
        self.transform = transform
        self.image_width = image_width
        self.image_height = image_height
        self.preprocessing_on = preprocessing_on
        self.load_on_demand = load_on_demand
        self.extract_bb_region = extract_bb_region

        preprocessed_path = './dataset/SFHDataset/SFH/preprocessed_data'
        dataset_path = os.path.join(preprocessed_path, dataset_source)

        if preprocessing_on and os.path.exists(dataset_path): # Load it if already generated
            with open(dataset_path, 'rb') as dataset_file:
                self.videos = pickle.load(dataset_file)
        else:
            self.hands_model = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            if load_on_demand:
                self.videos = []
                for label in os.listdir(self.video_path):
                    label_path = os.path.join(self.video_path, label)
                    for video_file in os.listdir(label_path):
                        video_path = os.path.join(label_path, video_file)
                        self.videos.append((video_path, int(label)))
            else:           
                self.videos = self.load_videos()
                if not os.path.exists(preprocessed_path):
                    os.makedirs(preprocessed_path)
                with open(dataset_path, 'wb') as dataset_file:
                    pickle.dump(self.videos, dataset_file)

    def load_videos(self, load_video_path = None):

        # If load_video_path is not None, then process a single video (for the on-demand option)
        if load_video_path is not None:
            # print("Processing video on-demand")
            cap = cv2.VideoCapture(load_video_path)

            # print(cap)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            regions = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.extract_bb_region:
                    # Preprocess the frame by extracting the bounding box (hand) region
                    hand_region = self.extract_hand_bb(frame, frame_width, frame_height, first_only=True) 
                    if hand_region is None:
                        hand_region = np.zeros((self.image_height, self.image_width, 3))
                    regions.append(hand_region)
                else:
                    # Simply resize the entire frame without extracting the bounding box
                    frame = cv2.resize(frame, (self.image_height, self.image_width))
                    regions.append(frame)

            cap.release()
            video = torch.stack([transforms.ToTensor()(region) for region in regions])
            if self.transform:
                video = self.transform(video)

            return video
        
        else: # otherwise, load all the videos (no-on-demand option)
            videos = []

            label_path_0 = os.path.join(self.video_path, '0')
            label_path_1 = os.path.join(self.video_path, '1')

            total_videos = len(os.listdir(label_path_0)) + len(os.listdir(label_path_1))

            pbar = tqdm(total=total_videos, desc="Preprocessing videos...", unit="item")

            for label in os.listdir(self.video_path):

                label_path = os.path.join(self.video_path, label)

                for video_file in os.listdir(label_path):

                    video_path = os.path.join(label_path, video_file)

                    cap = cv2.VideoCapture(video_path)

                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    regions = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        if self.extract_bb_region:
                            hand_region = self.extract_hand_bb(frame, frame_width, frame_height, first_only=True) 
                            if hand_region is None:
                                hand_region = np.zeros((self.image_height, self.image_width, 3))
                            regions.append(hand_region)
                        else:
                            frame = cv2.resize(frame, (self.image_height, self.image_width))
                            regions.append(frame)

                    cap.release()
                    video = torch.stack([transforms.ToTensor()(region) for region in regions])
                    if self.transform:
                        video = self.transform(video)

                    videos.append((video, int(label)))
                    pbar.update(1)

            return videos

    def extract_hand_bb(self, frame, frame_width, frame_height, first_only=True):
        # Apparently the process() function is not thread-safe. With multiple workers, the code gets stuck here.
        results = self.hands_model.process(frame)
        #print("Hands Detector Model is ending processing for this frame.")
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
        if self.load_on_demand:
            video_path, label = self.videos[index]
            video = self.load_videos(load_video_path=video_path)
            return video, label
        else:
            video, label = self.videos[index]
            return video, label

if __name__ == '__main__':
    video_path = "./SFH/SFH_Dataset_S2CITIES"
    dataset_name = "./"

    # Define any transforms you want to apply to the videos
    transform = transforms.Compose([
        UniformTemporalSubsample(num_samples=16, temporal_dim=0),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create the VideoDataset and DataLoader
    dataset = Signal4HelpDataset(video_path, 
                                 image_width=112, 
                                 image_height=112, 
                                 preprocessing_on=False,  # Do not load from already existing file
                                 load_on_demand=False,    # Do not load on-demand (preprocess all the videos)
                                 transform=transform)
    # Check that the dataset has correctly been created
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    for idx, batch in enumerate(dataloader):
        print(batch) 
        break


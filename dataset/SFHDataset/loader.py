import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
from pytorchvideo.transforms import UniformTemporalSubsample

class VideoDataset(Dataset):
    def __init__(self, video_path, image_width, image_height, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.image_width = image_width
        self.image_height = image_height
        self.videos = self.load_videos()

    def load_videos(self):
        videos = []
        for label in os.listdir(self.video_path):
            label_path = os.path.join(self.video_path, label)
            for video_file in os.listdir(label_path):
                video_path = os.path.join(label_path, video_file)
                cap = cv2.VideoCapture(video_path)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (self.image_width, self.image_height))
                    frames.append(frame)
                cap.release()
                video = torch.stack([transforms.ToTensor()(frame) for frame in frames])
                videos.append((video, int(label)))

        return videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video, label = self.videos[index]

        if self.transform:
            video = self.transform(video)

        return video, label

if __name__ == '__main__':
    # Set the video path
    video_path = "./SFH/SFH_Dataset_S2CITIES"

    # Define any transforms you want to apply to the videos
    transform = transforms.Compose([
        UniformTemporalSubsample(num_samples=16, temporal_dim=0),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create the VideoDataset and DataLoader
    dataset = VideoDataset(video_path, image_width=112, image_height=112, transform=transform)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate over the dataloader to get batches of videos and labels
    for batch_videos, batch_labels in dataloader:
        # Do something with the batch of videos and labels
        print(batch_videos.shape)
        print(batch_labels)
        # break


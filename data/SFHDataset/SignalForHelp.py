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

            # print(frame.size)
            clip.append(frame)

        cap.release()

        n_frames = len(clip)
        # Apply Temporal Transform
        if temporal_transform is not None:
            # print(f"Clip lenght: {n_frames}")
            clip = [Image.fromarray(frame) for frame in clip]
            frame_indices = list(range(1, n_frames+1))
            frame_indices = temporal_transform(frame_indices)

            print(frame_indices)
            clip = [clip[i-1] for i in frame_indices]
        else:
            # If there are less then sample_duration frames, repeat the last one
            if n_frames < sample_duration:
                num_black_frames = sample_duration - n_frames
                for _ in range(num_black_frames):
                    clip.append(np.zeros_like(clip[0]))
            # If there are more than sample_duration, take the ones in the middle
            else:
                start_idx = (n_frames-sample_duration) // 2
                end_idx = start_idx + sample_duration
                clip = clip[start_idx:end_idx]

            # Convert every frame to PIL Image
            clip = [Image.fromarray(frame) for frame in clip]

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
            video_path = ' '.join(fields[:-1])
            label = int(fields[-1])
            self.videos.append((video_path, label))

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
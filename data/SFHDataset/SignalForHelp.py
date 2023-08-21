import torch
from enum import Enum
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvideotransforms.volume_transforms import ClipToTensor
from torchvideotransforms.video_transforms import Compose, RandomHorizontalFlip, Resize, RandomResizedCrop, RandomRotation
import os
import cv2
import mediapipe as mp
from PIL import Image

class Signal4HelpDataset(Dataset):

    class FrameSelectStrategy(Enum):
        FROM_BEGINNING = 0
        FROM_END = 1
        RANDOM = 2

    class FramePadding(Enum):
        REPEAT_END = 0
        REPEAT_BEGINNING = 2

    def __init__(self, annotation_path, clip_transform=None, number_of_frames=16,
                 frame_select_strategy=FrameSelectStrategy.RANDOM, frame_padding=FramePadding.REPEAT_END,
                 downsampling=1):
        
        self.clip_transform=clip_transform
        self.number_of_frames=number_of_frames
        self.frame_select_strategy=frame_select_strategy
        self.frame_padding=frame_padding
        self.downsampling=downsampling
        self.videos = []

        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            fields = line.strip().split(' ')
            video_path = ' '.join(fields[:-1])
            label = int(fields[-1])
            self.videos.append((video_path, label))

    def _add_padding(self, frames: list, number_of_frames: int, downsampling: int, frame_padding: FramePadding):
        difference = number_of_frames * downsampling - len(frames)
        if difference > 0:
            if frame_padding == self.FramePadding.REPEAT_BEGINNING:
                frame_index_to_repeat = 0
            elif frame_padding == self.FramePadding.REPEAT_END:
                frame_index_to_repeat = -1
            else:
                raise ValueError("Frame Padding Type not supported")

            frames += [frames[frame_index_to_repeat] for _ in range(difference)]
        return frames

    def _select_frames(self, frames: list, frame_select_strategy: FrameSelectStrategy, number_of_frames: int, downsampling: int):
        if len(frames) <= number_of_frames * downsampling:
            return [frames[idx] for idx in list(range(0, len(frames), downsampling))]
        else:
            if frame_select_strategy == self.FrameSelectStrategy.FROM_BEGINNING:
                return [frames[idx] for idx in list(range(0, number_of_frames*downsampling, downsampling))]
                # return frames[:number_of_frames]
            elif frame_select_strategy == self.FrameSelectStrategy.FROM_END:
                return [frames[idx] for idx in list(range(len(clip)-(number_of_frames * downsampling), len(clip), downsampling))]
                # return frames[-number_of_frames:]
            elif frame_select_strategy == self.FrameSelectStrategy.RANDOM:
                difference = len(frames) - (number_of_frames * downsampling)
                random_start_index = torch.randint(0, difference, (1,)).item()
                end_index = random_start_index + (number_of_frames * downsampling)
                return [frames[idx] for idx in list(range(random_start_index, end_index, downsampling))]
                # return frames[random_start_index:end_index]
            else:
                raise ValueError("FrameSelectStrategy not supported.")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_path, label = self.videos[index]

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
        if len(clip) == 0:
            raise FileNotFoundError(f"ERROR: Could not find or open video at path {video_path}.")
        clip = self._add_padding(frames=clip, number_of_frames=self.number_of_frames,
                                 downsampling=self.downsampling, frame_padding=self.frame_padding)
        clip = self._select_frames(frames=clip, frame_select_strategy=self.frame_select_strategy,
                                   number_of_frames=self.number_of_frames, downsampling=self.downsampling)
        clip = [Image.fromarray(frame).convert('RGB') for frame in clip]

        if self.clip_transform:
            clip = self.clip_transform(clip)

        return clip, label

if __name__ == '__main__':

    frame_size = 112
    clip_duration = 16

    test_clip_transform = Compose([
        Resize(size=(frame_size, frame_size, 3)), # Resize any frame to shape (112, 112, 3) (H, W, C)
        ClipToTensor()
    ])  
    # Test this script from the repo root directory (python data/SFHDataset/SignalForHelp.py)
    dataset = Signal4HelpDataset(annotation_path='data/SFHDataset/test_annotations.txt', clip_transform=test_clip_transform,
                                 number_of_frames=clip_duration)
    clip, label = dataset[0]
    print(clip.shape) # Expected [3, 16, 112, 112]
import json
import os
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

def load_videos(annotation_path = 'dataset/WLASL/WLASL_v0.3.json', 
                missing_path = 'dataset/WLASL/missing.txt',
                label_path = 'dataset/WLASL/wlasl_class_list.txt'):

    splits = {
        'train': [],
        'val': [],
        'test': []
    }

    with open(missing_path, 'r') as f:
        missings = f.readlines()

    missings = [idx.strip() for idx in missings]

    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    label_idx = {}
    
    with open(label_path, 'r') as f:
        labels = f.readlines()
        for idx_name in labels:
            idx, name = idx_name.strip().split('\t')
            label_idx[name] = int(idx)

    for gloss in data:
        label = gloss['gloss']
        instances = gloss['instances']
        for instance in instances:
            split = instance['split']
            video_id = instance['video_id']
            if video_id in missings:
                continue
            fps = int(instance['fps'])
            bbox = instance['bbox']
            frame_start, frame_end = int(instance['frame_start']), int(instance['frame_end'])
            splits[split].append({
                'video_path': os.path.join('dataset/WLASL/videos', f"{video_id}.mp4"),
                'fps': fps,
                'bbox': bbox,
                'frame_start': frame_start, 'frame_end': frame_end,
                'label': label_idx[label]
            })

    return splits


class WLASL(Dataset):
    def __init__(self, videos):
        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        video_path = video['video_path']

        print(video_path)
        label = video['label']

        clip = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                clip.append(frame)
            else:
                break

        cap.release()

        # because the JSON file indexes from 1.
        idx_start = video['frame_start'] - 1
        idx_end = video['frame_end'] - 1
        xmin, ymin, xmax, ymax = video['bbox'] 

        clip = clip[idx_start: idx_end+1]
        clip = [frame[ymin:ymax+1, xmin:xmax+1] for frame in clip]

        return clip, label


if __name__ == '__main__':
    splits = load_videos()

    train_dataset = WLASL(splits['train'])

    # Test if cropping and frame selection is correctly done.

    clip, label = train_dataset[0]
    # print(clip)
    print(len(clip))
    
    size = clip[0].shape[:2][::-1]

    out = cv2.VideoWriter('test_WLASL/1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps=25, frameSize=size)

    for i in range(len(clip)):
        # writing to a image array
        out.write(clip[i])
    out.release()

    clip, label = train_dataset[1]

    # print(clip)
    print(len(clip))

    size = clip[0].shape[:2][::-1]

    out = cv2.VideoWriter('test_WLASL/2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps=25, frameSize=size)

    for i in range(len(clip)):
        # writing to a image array
        out.write(clip[i])
    out.release()

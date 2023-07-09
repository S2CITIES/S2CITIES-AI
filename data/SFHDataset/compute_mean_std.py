from SignalForHelp import load_video
import torch
import json

def get_SFH_mean_std(image_height=112, image_width=112, n_frames=16):

    info_file = 'data/SFHDataset/info.json'

    with open(info_file, 'r') as file:
        info = json.load(file)

    if info["mean"] == [-1, -1, -1] and info["std"] == [-1, -1, -1]:
        n_samples = 0
        channel_sum = 0
        channel_squared_sum = 0

        with open('data/SFHDataset/train_annotations.txt', 'r') as annotation_file:
            lines = annotation_file.readlines()
        
        for line in lines:
            video_path, _ = line.strip().split(' ')
            video = load_video(video_path, image_height=image_height, image_width=image_width) # video shape TCHW
            channel_sum += torch.sum(video, dim=(0, 2, 3))
            channel_squared_sum += torch.sum(video**2, dim=(0, 2, 3))
            n_samples += 1
        
        mean = channel_sum / (n_samples*n_frames*image_height*image_width)
        std = torch.sqrt((channel_squared_sum / (n_samples*n_frames*image_height*image_width)) - (mean ** 2))
        info["mean"] = mean.tolist()
        info["std"] = std.tolist()

        with open(info_file, 'w') as file:
            json.dump(info, file, indent=4)  # indent=4 for pretty formatting
    
    else:
        mean = info["mean"]
        std = info["std"]  

    return mean, std

get_SFH_mean_std()
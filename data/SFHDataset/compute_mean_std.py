from data.SFHDataset.SignalForHelp import load_video
from transforms.spatial_transforms import ToTensor, Compose, Scale
import torch
import json
from tqdm import tqdm

def get_SFH_mean_std(target_dataset, image_size=112, norm_value=1.0, force_compute=False):

    info_file = 'data/SFHDataset/info.json'

    with open(info_file, 'r') as file:
        info = json.load(file)

    if (not info[target_dataset]["mean"] or not info[target_dataset]["std"]) or force_compute:
        n_frames = 0
        channel_sum = 0
        channel_squared_sum = 0

        with open('data/SFHDataset/train_annotations.txt', 'r') as annotation_file:
            lines = annotation_file.readlines()
        
        for line in tqdm(lines):
            video_path, _ = line.strip().split(' ')

            spatial_transform = Compose([
                Scale(image_size),
                ToTensor(norm_value)
            ])

            video = load_video(video_path, spatial_transform=spatial_transform) # video shape CTHW
            channel_sum += torch.sum(video, dim=(1, 2, 3))
            channel_squared_sum += torch.sum(video**2, dim=(1, 2, 3))
            n_frames += video.shape[-3]
        
        mean = channel_sum / (n_frames*image_size*image_size)
        std = torch.sqrt((channel_squared_sum / (n_frames*image_size*image_size)) - (mean ** 2))
        info[target_dataset]["mean"] = mean.tolist()
        info[target_dataset]["std"] = std.tolist()

        with open(info_file, 'w') as file:
            json.dump(info, file, indent=4)  # indent=4 for pretty formatting
    
    else:
        mean = info[target_dataset]["mean"]
        std = info[target_dataset]["std"]  

    return mean, std

if __name__ == '__main__':
    get_SFH_mean_std(force_compute=True)
from data.SFHDataset.SignalForHelp import load_video
from transforms.spatial_transforms import ToTensor, Compose, Scale
import torch
import json
from tqdm import tqdm

def get_SFH_min_max(target_dataset, image_size=112, norm_value=1.0, force_compute=True):

    info_file = 'data/SFHDataset/info.json'

    with open(info_file, 'r') as file:
        info = json.load(file)
        min = 0
        max = 255

    if force_compute:
        channel_max = 0
        channel_min = 255

        with open('data/SFHDataset/train_annotations.txt', 'r') as annotation_file:
            lines = annotation_file.readlines()
        
        for line in tqdm(lines, ascii=True, desc='Computing Mean/Std.'):
            video_path, _ = line.strip().split(' ')

            spatial_transform = Compose([
                Scale(image_size),
                ToTensor(norm_value)
            ])

            video = load_video(video_path, spatial_transform=spatial_transform) # video shape CTHW
            if channel_max<torch.max(video, dim=(1, 2, 3)):
                channel_max = torch.max(video, dim=(1, 2, 3))
            if channel_min>torch.min(video, dim=(1, 2, 3)):
                channel_min = torch.min(video, dim=(1, 2, 3))

        #info[target_dataset]["max"] = channel_max.tolist()
        #info[target_dataset]["min"] = channel_min.tolist()

        with open(info_file, 'w') as file:
            json.dump(info, file, indent=4)  # indent=4 for pretty formatting

        min = channel_min
        max = channel_max

    return min, max

if __name__ == '__main__':
    get_SFH_min_max(force_compute=True)
from data.SFHDataset.SignalForHelp import load_video
from transforms.spatial_transforms import ToTensor, Compose, Scale
import torch
import json
from tqdm import tqdm

def get_SFH_min_max(target_dataset, image_size=112, norm_value=1.0, force_compute=True):

    info_file = 'data/SFHDataset/info.json'

    with open(info_file, 'r') as file:
        info = json.load(file)
        min = [0, 0, 0]
        max = [1, 1, 1]
        print(f"before if id: {id(min)}")


    if force_compute:
        channel_max = [0, 0, 0]
        channel_min = [255, 255, 255]

        with open('data/SFHDataset/train_annotations.txt', 'r') as annotation_file:
            lines = annotation_file.readlines()
        
        for line in tqdm(lines, ascii=True, desc='Computing Mean/Std.'):
            #video_path, _ = line.strip().split(' ')
            video_path = line.strip()[:-2]

            spatial_transform = Compose([
                Scale(image_size),
                ToTensor(norm_value)
            ])

            video = load_video(video_path, spatial_transform=spatial_transform) # video shape CTHW
            mx = torch.amax(video, dim=(1, 2, 3))
            mn = torch.amin(video, dim=(1, 2, 3))
            print(channel_max)
            print(mn)
            for c in range(len(mx)):
                if channel_max[c]<mx[c]:
                    channel_max = mx[c]
                if channel_min[c]>mn[c]:
                    channel_min = mn[c]

        #info[target_dataset]["max"] = channel_max.tolist()
        #info[target_dataset]["min"] = channel_min.tolist()

        with open(info_file, 'w') as file:
            json.dump(info, file, indent=4)  # indent=4 for pretty formatting

        min = channel_min
        max = channel_max
        print(f"Inside if id: {id(min)}")
    
    print(f"after if id: {id(min)}")

    return min, max

if __name__ == '__main__':
    get_SFH_min_max(force_compute=True)
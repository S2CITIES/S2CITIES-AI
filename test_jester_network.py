# Testing how the pre-trained model work on Jester dataset
from build_models import build_model
import torch
import torch.nn
import os
import cv2
from PIL import Image
from data.SFHDataset.SignalForHelp import load_video
import transforms.spatial_transforms as SPtransforms
import transforms.temporal_transforms as TPtransforms

if __name__ == '__main__':
    # Load Jester video (Test video 1417)
    video_path = './dataset/Jester_videos/594.mp4'

    test_temporal_transform = TPtransforms.TemporalRandomCrop(16, 2)
    # Initialize spatial and temporal transforms (test versions)
    test_spatial_transform = SPtransforms.Compose([
        SPtransforms.Scale(112),
        SPtransforms.CenterCrop(112), # Central Crop in Test
        SPtransforms.ToTensor(norm_value=1.0),
        SPtransforms.Normalize(mean=[0,0,0], std=[1,1,1])
    ])

    clip = load_video(video_path=video_path,
                      temporal_transform=test_temporal_transform,
                      spatial_transform=test_spatial_transform,
                      sample_duration=16,
                      norm_value=1.0,
                      save_output=True)

    clip = clip.unsqueeze(dim=0)

    print(clip.shape)
    # Load MobileNetv2 Network
    base_model_path = './models/pretrained/jester/jester_mobilenetv2_1.0x_RGB_16_best.pth'
    model = build_model(model_path=base_model_path, 
                    type='mobilenetv2', 
                    gpus=[0],
                    sample_size=112,
                    output_features=27,
                    sample_duration=16,
                    num_classes=27,
                    finetune=False)
    
    with torch.no_grad():
        logits = model(clip)
        print(logits.shape)
        output = torch.softmax(logits, dim=1)
        print(output)

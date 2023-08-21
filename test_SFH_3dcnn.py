import os
import json
import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from data.SFHDataset.SignalForHelp import Signal4HelpDataset
from build_models import build_model
import numpy as np
import functools
from tqdm import tqdm
from train_args import parse_args
from torchvideotransforms.volume_transforms import ClipToTensor
from torchvideotransforms.video_transforms import Compose, RandomHorizontalFlip, Resize, RandomResizedCrop, RandomRotation

## NOTE: This script tests a 3D-CNN model trained on the SFH task, by reporting Video Accuracy.
## Note that Video Accuracy and Clip Accuracy are two different things!

# Silent warnings about TypedStorage deprecations that appear on the cluster
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

args = parse_args()

def compute_video_accuracy(ground_truth, predictions):
    # Inspired by evaluation performed in Karpathy et al. CVPR14
    # Other evaluations are also possible

    # ground_truth: df with fields ['video-id', 'label']
    # predictions: df with fields ['video-id', 'label', 'score']
    # Takes the first top-k predicted labels (in ascending order), compare them with the ground-truth labels
    # and compute the average number of hits per video.
    # Number of hits = Number of steps in which one the top-k predicted labels is equal to the ground-truth.

    video_ids = np.unique(ground_truth['video-id'].values)
    avg_hits_per_video = np.zeros(video_ids.size)
    
    for i, video in enumerate(video_ids):
        pred_idx = predictions['video-id'] == video
        if not pred_idx.any():
            continue
        # Get prediction scores
        this_pred = predictions.loc[pred_idx].reset_index(drop=True)
        print(this_pred)
        sort_idx = this_pred['score'].values.argsort()[::-1][:1]    # Take the label with the highest predicted score
        this_pred = this_pred.loc[sort_idx].reset_index(drop=True)
        pred_label = this_pred['label'].tolist()
        print(pred_label)
        # Get ground truth label for video with video-id 
        gt_idx = ground_truth['video-id'] == video
        gt_label = ground_truth.loc[gt_idx]['label'].tolist()
        avg_hits_per_video[i] = np.mean([1 if this_label in pred_label else 0
                                         for this_label in gt_label])
        
    return float(avg_hits_per_video.mean())

def test(videos, model, num_frames_per_clip, downsampling_in_clip, clip_transform, pbar):
    # gt = []
    # preds = []
    corrects = 0
    totals = 0

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(videos):
            video, info = data
            gt_label = info['label']
            # gt.append({
            #     'video-id': info['video-id'],
            #     'label': info['label']
            # })

            video_logits = []

            for j in range(0, len(video), num_frames_per_clip):
                clip = video[j:j+num_frames_per_clip]
                if len(clip) < num_frames_per_clip:
                    difference = len(clip) - num_frames_per_clip
                    clip += [clip[-1] for _ in range(difference)] # Repeat the last frame if the last chunk has size < number_of_frames
                if clip_transform:
                    clip = clip_transform(clip)

                clip = clip.float()
                # Add the batch dim to video
                clip = clip.unsqueeze(dim=0)
                clip = clip.to(device)

                logits = model(clip)
                video_logits.append(logits)

                # scores = torch.softmax(logits, dim=1).cpu().view(-1)
                # for label, score in enumerate(scores):
                #     preds.append({
                #         'video-id': info['video-id'], 
                #         'label': label, 
                #         'score': score.item()
                #         })
            
            video_logits = torch.stack(video_logits, dim=0)
            video_logits = torch.mean(video_logits, dim=0)
            video_scores = torch.softmax(video_logits, dim=1)
            pred_label = torch.argmax(video_scores, dim=1)

            totals += 1
            corrects += (pred_label.item() == gt_label)*1
            pbar.update(1)
    
    # gt = pd.DataFrame(gt)
    # preds = pd.DataFrame(preds)
    # video_accuracy = compute_video_accuracy(ground_truth=gt, predictions=preds)
    video_accuracy = 100 * (corrects/totals)
    print(f"Average Video Accuracy on Test Set: {video_accuracy}")
    return video_accuracy

if __name__ == '__main__':

    batch_size=args.batch
    num_epochs=args.epochs
    dataset_name = args.data_path.split("/")[-1]

    frame_size = args.sample_size
    clip_duration = args.sample_duration

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device {}".format(device))

    # Initialize spatial clip transforms (test versions)
    test_clip_transform = Compose([
        Resize(size=(frame_size, frame_size, 3)), # Resize any frame to shape (112, 112, 3) (H, W, C)
        ClipToTensor()
    ])  
    
    test_dataset = Signal4HelpDataset(os.path.join(args.annotation_path, 'test_annotations.txt'), test_on_videos=True)
    
    print('Size of Test Set: {}'.format(len(test_dataset)))

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    model = build_model(model_path='checkpoints/best_model_finetune-mobilenetv2-sfh-wcrossentropy.h5', 
                        type='mobilenetv2',
                        num_classes=2, # Number of classes of the original pre-trained model on Jester dataset 
                        gpus=list(range(0, num_gpus)),
                        sample_size=112,
                        sample_duration=16,
                        output_features=2,
                        finetune=False,      # Fine-tune the classifier (last fully connected layer)
                        state_dict=True)     # If only the state_dict was saved

    # Initialize tqdm progress bar for tracking training steps
    pbar = tqdm(total=len(test_dataset))
    pbar.set_description('Testing on Test Videos')
    test(videos=test_dataset, model=model, num_frames_per_clip=16, downsampling_in_clip=1, clip_transform=test_clip_transform, pbar=pbar)
    
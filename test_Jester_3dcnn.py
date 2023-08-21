import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from data.Jester.jesterdataset.jester_dataset import JesterDataset
from build_models import build_model
import numpy as np
from tqdm import tqdm
from train_args import parse_args
from torchvideotransforms.volume_transforms import ClipToTensor
from torchvideotransforms.video_transforms import Compose, RandomHorizontalFlip, Resize, RandomResizedCrop, RandomRotation

# Silent warnings about TypedStorage deprecations that appear on the cluster
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

args = parse_args()

def compute_video_accuracy(ground_truth, predictions, top_k=3):
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
        this_pred = predictions.loc[pred_idx].reset_index(drop=True)
        # Get top K predictions sorted by decreasing score.
        sort_idx = this_pred['score'].values.argsort()[::-1][:top_k]
        this_pred = this_pred.loc[sort_idx].reset_index(drop=True)
        # Get top K labels and compare them against ground truth.
        pred_label = this_pred['label'].tolist()
        gt_idx = ground_truth['video-id'] == video
        gt_label = ground_truth.loc[gt_idx]['label'].tolist()
        avg_hits_per_video[i] = np.mean([1 if this_label in pred_label else 0
                                         for this_label in gt_label])
        
    return float(avg_hits_per_video.mean())

def compute_clip_accuracy(logits, labels, topk=(1,)):
    batch_size = labels.size(0)
    _, topk_preds = torch.softmax(logits, dim=1).topk(max(topk), 1, True, True)
    topk_preds = topk_preds.t()
    corrects = topk_preds.eq(labels.view(1, -1).expand_as(topk_preds)) 
    res = []
    for k in topk:
        corrects_k = corrects[:k].reshape(-1).float().sum(0)
        res.append(corrects_k.mul_(100.0 / batch_size))
    return res

# NOTE: Models trained on Jester cannot be evaluated right now because I'm missing test labels...

def test(loader, model, pbar, device):
    totals = 0
    top1 = []
    top5 = []
    video_results = []
    with torch.no_grad():
        model.eval()

        for i, data in enumerate(loader):
            clips, labels = data
            clips = clips.float()
            clips = clips.to(device)
            labels = labels.to(device)

            logits = model(clips)
            acc1, acc5 = compute_clip_accuracy(logits=logits, labels=labels, topk=(1,5))
            
            totals += clips.shape[0]
            top1.append((acc1, clips.shape[0]))
            top5.append((acc5, clips.shape[0]))

            pbar.update(clips.shape[0])

    top1_accuracy = 0
    top5_accuracy = 0
    for idx, _ in enumerate(top1):
        top1_accuracy += top1[idx][0] * top1[idx][1]
        top5_accuracy += top5[idx][0] * top5[idx][1]
    
    avg_top1_accuracy = top1_accuracy / totals
    avg_top5_accuracy = top5_accuracy / totals

    print('Test Top1 Clip Accuracy: {:.2f}%, Top5 Clip Accuracy: {:.2f}'.format(avg_top1_accuracy, avg_top5_accuracy))

if __name__ == '__main__':

    batch_size=args.batch
    num_epochs=args.epochs

    clip_duration = args.sample_duration
    frame_size = args.sample_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device {}".format(device))

    # Initialize spatial and temporal transforms (test versions)
    test_clip_transform = Compose([
        Resize(size=(frame_size, frame_size, 3)), # Resize any frame to shape (112, 112, 3) (H, W, C)
        ClipToTensor()
    ])

    # Test again with Random Crops on clips from the test set.
    # NOTE: Pay attention - both in train, test and validation, Clip accuracy is computed, not Video accuracy.
    # To compute Video accuracy, we need to traverse the entire video while extracting clips and doing inference with our models.
    # Then, predicted logits should be averaged and the results after the softmax layer should be used to make a prediction 
    # for the entire video.
    # TODO: A good idea would be to plot the logits (better class predictions) produced at each step, and see how
    # they evolve during time. 

    test_set = JesterDataset(csv_file='data/Jester/jester_data/Test.csv',
                              video_dir='data/Jester/jester_data/Test',
                              number_of_frames=clip_duration, 
                              video_transform=test_clip_transform)

    print('Size of Test Set: {}'.format(len(test_set)))

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = 27
    model = build_model(model_path='checkpoints/best_model_jester-mobilenetv2-singlegpu.h5', 
                        type=args.model,
                        num_classes=num_classes,
                        gpus=list(range(0, num_gpus)),
                        sample_size=args.sample_size,
                        sample_duration=args.sample_duration,
                        output_features=num_classes,
                        finetune=False)

    # Initialize tqdm progress bar for tracking test steps
    pbar = tqdm(total=len(test_set))
    pbar.set_description("[Testing]")

    test(loader=test_dataloader, 
         model=model,
         pbar=pbar,
         device=device)
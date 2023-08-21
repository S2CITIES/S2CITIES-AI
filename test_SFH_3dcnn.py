import os
import json
import torch
import torch.nn as nn
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

# Using wanbd (Weights and Biases, https://wandb.ai/) for run tracking
import wandb

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

def test(loader, model, criterion, device, epoch=None):
    totals = 0
    corrects = 0
    y_pred = []
    y_true = []
    val_loss = []

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(loader):
            videos, labels = data
            videos = videos.float()
            videos = videos.to(device)

            logits = model(videos)

            labels = labels.to(device)
            val_loss_batch = criterion(logits, labels)

            val_loss.append(val_loss_batch.item())

            y_true.append(labels)

            y_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            corrects += (y_preds == labels).sum().item()
            totals += y_preds.shape[0]

            y_preds = y_preds.detach().cpu()
            y_pred.append(y_preds)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    val_accuracy = 100 * corrects / totals
    val_loss = np.array(val_loss).mean()

    print('Test Video Accuracy: {:.2f}%'.format(val_accuracy))

    return val_accuracy, val_loss

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
    
    test_dataset = Signal4HelpDataset(os.path.join(args.annotation_path, 'test_annotations.txt'), 
                                clip_transform=test_clip_transform,
                                number_of_frames=clip_duration,
                                downsampling=args.downsampling)
    print('Size of Test Set: {}'.format(len(test_dataset)))


    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # Initialize DataLoaders
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # User provided entire path for pre-trained weights
    base_model_path = args.pretrained_path 

    model = build_model(model_path=base_model_path, 
                        type=args.model,
                        num_classes=27, # Number of classes of the original pre-trained model on Jester dataset 
                        gpus=list(range(0, num_gpus)),
                        sample_size=args.sample_size,
                        sample_duration=args.sample_duration,
                        output_features=args.output_features,
                        finetune=True,      # Fine-tune the classifier (last fully connected layer)
                        state_dict=True)    # If only the state_dict was saved
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)

    if args.nesterov:
        args.dampening = 0.
    
    # classifier = model.module.get_submodule('classifier')
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(list(model.parameters()), 
                                                    lr=args.lr, 
                                                    momentum=args.momentum, 
                                                    dampening=args.dampening,
                                                    weight_decay=args.wd,
                                                    nesterov=args.nesterov)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=args.lr_patience, factor=0.1)

    if args.output_features == 1:
        # NOTE: nn.BCEWithLogitsLoss already contains the Sigmoid layer inside
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=cross_entropy_weights)

    # Initialize tqdm progress bar for tracking training steps
    pbar = tqdm(total=len(train_dataset))

    # Create model saves path if it doesn't exist yet
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    train(model=model, 
          optimizer=optimizer,
          scheduler=scheduler,
          criterion=criterion, 
          train_loader=train_dataloader,
          val_loader=val_dataloader, 
          num_epochs=num_epochs,
          output_features=args.output_features, 
          device=device, 
          pbar=pbar)

    # Load the best checkpoint obtained until now
    best_checkpoint=torch.load(os.path.join(args.model_save_path, f'best_model_{args.exp}.h5'))
    model.load_state_dict(best_checkpoint)
  
    test(loader=test_dataloader, 
         model=model,
         criterion=criterion,
         output_features=args.output_features,
         device=device)
    
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from data.SFHDataset.SignalForHelp import Signal4HelpDataset, load_video
from build_models import build_model
import numpy as np
import functools
from tqdm import tqdm
from train_args import parse_args
import transforms.spatial_transforms as SPtransforms
import transforms.temporal_transforms as TPtransforms
from data.SFHDataset.compute_mean_std import get_SFH_mean_std

# Using wanbd (Weights and Biases, https://wandb.ai/) for run tracking

# Silent warnings about TypedStorage deprecations that appear on the cluster
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

args = parse_args()

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

    if epoch is not None:
        # Save metrics with wandb
        wandb.log({"val_accuracy": val_accuracy, "val_loss": val_loss}, commit=True)

        print('[Epoch {}] Validation Accuracy: {:.2f}%'.format(epoch, val_accuracy))
    else:
        wandb.log({"test_accuracy": val_accuracy, "test_loss": val_loss}, commit=True)

        print('Test Accuracy: {:.2f}%'.format(val_accuracy))

    return val_accuracy, val_loss

if __name__ == '__main__':

    batch_size=args.batch
    num_epochs=args.epochs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device {}".format(device))

    # No erandom scaling - Just original scale
    args.scales = [1.]

    # Initialize spatial and temporal transforms (training versions)
    if args.train_crop == 'random':
        crop_method = SPtransforms.MultiScaleRandomCrop(args.scales, args.sample_size)
    elif args.train_crop == 'corner':
        crop_method = SPtransforms.MultiScaleCornerCrop(args.scales, args.sample_size)
    elif args.train_crop == 'center':
        crop_method = SPtransforms.MultiScaleCornerCrop(args.scales, args.sample_size, crop_positions=['c'])
    
    if not args.no_norm:
        target_dataset = args.data_path.split('/')[-1]
        # Compute channel-wise mean and std. on the training set
        mean, std = get_SFH_mean_std(target_dataset=target_dataset,
                                    image_size=args.sample_size, 
                                    norm_value=args.norm_value, 
                                    force_compute=args.recompute_mean_std)
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]

    train_spatial_transform = SPtransforms.Compose([
        SPtransforms.RandomHorizontalFlip(),
        crop_method,
        SPtransforms.ToTensor(args.norm_value),
        SPtransforms.Normalize(mean=mean, std=std)
    ])

    train_temporal_transform = None
    if args.temp_transform:
        train_temporal_transform = TPtransforms.TemporalRandomCrop(args.sample_duration, args.downsample)

    # Initialize spatial and temporal transforms (validation versions)
    val_spatial_transform = SPtransforms.Compose([
        SPtransforms.Scale(args.sample_size),
        SPtransforms.CenterCrop(args.sample_size),
        SPtransforms.ToTensor(args.norm_value),
        SPtransforms.Normalize(mean=mean, std=std)
    ])

    val_temporal_transform = None
    if args.temp_transform:
        val_temporal_transform = TPtransforms.TemporalCenterCrop(args.sample_duration, args.downsample)

    # Initialize spatial and temporal transforms (test versions)
    test_spatial_transform = SPtransforms.Compose([
        SPtransforms.Scale(args.sample_size),
        SPtransforms.CornerCrop(args.sample_size, crop_position='c'), # Central Crop in Test
        SPtransforms.ToTensor(args.norm_value),
        SPtransforms.Normalize(mean=mean, std=std)
    ])

    test_temporal_transform = None
    if args.temp_transform:
        test_temporal_transform = TPtransforms.TemporalRandomCrop(args.sample_duration, args.downsample)

 
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    if args.pretrained_path == 'auto':
        # 'Build' path for pretrained weights with provided information
        if args.model in ['mobilenet', 'mobilenetv2']:
            base_model_path='models/pretrained/jester/jester_{model}_1.0x_RGB_16_best.pth'.format(model=args.model)
        else:
            base_model_path='models/pretrained/jester/jester_squeezenet_RGB_16_best.pth'
    else:
        # User provided entire path for pre-trained weights
        base_model_path = args.pretrained_path 

    cam_model = build_model(model_path=base_model_path, 
                type="CAM", 
                gpus=list(range(0, num_gpus)),
                sample_size=args.sample_size,
                sample_duration=args.sample_duration,
                finetune=True)
    print(f"Total parameters: {sum(p.numel() for p in cam_model.parameters())}")
    trainable_params = sum(p.numel() for p in cam_model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)

    if args.nesterov:
        args.dampening = 0.

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(list(cam_model.parameters()), 
                                                    lr=args.lr, 
                                                    momentum=args.momentum, 
                                                    dampening=args.dampening,
                                                    weight_decay=args.wd,
                                                    nesterov=args.nesterov)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(list(cam_model.parameters()), lr=args.lr, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=args.lr_patience, factor=0.1)

    criterion = nn.CrossEntropyLoss()

    # Load the best checkpoint obtained until now
    best_checkpoint=torch.load(os.path.join(args.model_save_path, f'best_model_{args.exp}.h5'))
    cam_model.load_state_dict(best_checkpoint)

    # LOAD 1 VIDEO
    video0 = load_video("'../gdrive/MyDrive/DRIVE S2CITIES/Artificial Intelligence/SFH_Dataset_S2CITIES/SFH_Dataset_S2CITIES_raw_extended_negatives_ratio1_224x224/0/vid_00817_00133.mp4", 
                    temporal_transform=test_temporal_transform,
                    spatial_transform=test_spatial_transform,
                    save_output=False)
    video1 = load_video("'../gdrive/MyDrive/DRIVE S2CITIES/Artificial Intelligence/SFH_Dataset_S2CITIES/SFH_Dataset_S2CITIES_raw_extended_negatives_ratio1_224x224/1/vid_00246_00009.mov", 
                    temporal_transform=test_temporal_transform,
                    spatial_transform=test_spatial_transform,
                    save_output=False)
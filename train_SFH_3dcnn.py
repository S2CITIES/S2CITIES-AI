import os
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
import transforms.spatial_transforms as SPtransforms
import transforms.temporal_transforms as TPtransforms
from data.SFHDataset.compute_mean_std import get_SFH_mean_std

# Using wanbd (Weights and Biases, https://wandb.ai/) for run tracking
import wandb

# Silent warnings about TypedStorage deprecations that appear on the cluster
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

args = parse_args()

# "Collate" function for our dataloaders
# def collate_fn(batch, transform):
#     # NOTE (IMPORTANT): Normalize (pytorchvideo.transforms) from PyTorchVideo wants a volume with shape CTHW, 
#     # which is then internally converted to TCHW, processed and then again converted to CTHW.
#     # Because our volumes are in the shape TCHW, we convert them to CTHW here, instead of doing it inside the training loop.

#     videos = [transform(video.permute(1, 0, 2, 3)) for video, _ in batch]
#     labels = [label for _, label in batch]

#     videos = torch.stack(videos)
#     labels = torch.tensor(labels)
#     return videos, labels

# TODO: Implement custom scheduler to manual adjust learning rate after N epochs
def train(model, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs, device, pbar=None):

    # Set up early stopping criteria
    patience = args.early_stop_patience
    min_delta = 0.001  # Minimum change in validation loss to be considered as improvement
    best_loss = float('inf')  # Initialize the best validation loss
    best_accuracy = 0.0
    counter = 0  # Counter to keep track of epochs without improvement

    ############### Training ##################
    for epoch in range(num_epochs):
        model.train()

        epoch_loss = []
        corrects = 0
        totals = 0

        if pbar:
            pbar.set_description("[Epoch {}]".format(epoch))
            pbar.reset()

        for i, data in enumerate(train_loader):
            videos, labels = data
            videos = videos.float()
            videos = videos.to(device) # Send inputs to CUDA

            logits = model(videos)

            labels = labels.to(device)

            loss = criterion(logits, labels)
            epoch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if pbar:
                pbar.update(videos.shape[0])

            y_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            corrects += (y_preds == labels).sum().item()
            totals += y_preds.shape[0]

        torch.save(model.state_dict(), os.path.join(args.model_save_path, f'model_{args.exp}_epoch_{epoch}.h5'))

        avg_train_loss = np.array(epoch_loss).mean()
        train_accuracy = 100 * corrects / totals

        print("[Epoch {}] Avg Loss: {}".format(epoch, avg_train_loss))
        print("[Epoch {}] Train Accuracy {:.2f}%".format(epoch, train_accuracy))

        # commit = false because I want commit to happen after validation (so that the step is incremented once per epoch)
        wandb.log({"train_accuracy": train_accuracy, "train_loss": avg_train_loss}, commit=False)

        # NOTE: test function validates the model when it takes in input the loader for the validation set
        val_accuracy, val_loss = test(loader=val_loader, model=model, criterion=criterion, device=device, epoch=epoch)
        scheduler.step(val_loss)

        # Checking early-stopping criteria
        if val_loss + min_delta < best_loss:
            best_loss = val_loss
            best_accuracy = val_accuracy
            counter = 0 # Reset the counter since there is improvement
            # Save the improved model
            torch.save(model.state_dict(),  os.path.join(args.model_save_path, f'best_model_{args.exp}.h5'))
        else:
            counter += 1 # Increment the counter, since there is no improvement

        # Check if training should be stopped 
        if counter >= patience:
            print(f"Early-stopping the training phase at epoch {epoch}")
            break
    
    print("--- END Training. Results - Best Val. Loss: {:.2f}, Best Val. Accuracy: {:.2f}".format(best_loss, best_accuracy))

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

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.wandb_project,
        entity=args.wandb_team,
        name=args.exp,  
        # track hyperparameters and run metadata
        config={
        "optimizer": args.optimizer,
        "lr": args.lr,
        "lr_patience": args.lr_patience,
        "momentum": args.momentum,
        "dampening": args.dampening if not args.nesterov else 0.,
        "nesterov": args.nesterov,
        "weight_decay": args.wd, 
        "architecture": args.model,
        "dataset": args.data_path.split("/")[-1],
        "epochs": num_epochs,
        "batch": batch_size,
        "sample_size": args.sample_size,
        "sample_duration": args.sample_duration,
        "train_crop": args.train_crop,
        "early_stop_patience": args.early_stop_patience,
        "no_norm": args.no_norm
        }
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device {}".format(device))

    # Set torch manual seed for reproducibility
    torch.manual_seed(args.manual_seed)

    # Init different scales for random scaling
    # args.scales = [args.initial_scale]
    # for i in range(1, args.n_scales):
    #     args.scales.append(args.scales[-1] * args.scale_step)

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

    # Log normalization mean and std for future reference
    wandb.log({"norm_mean": mean, "norm_std": std})
    
    print(f"Train mean: {mean}")
    print(f"Train std.: {std}")

    train_spatial_transform = SPtransforms.Compose([
        SPtransforms.RandomHorizontalFlip(),
        crop_method,
        SPtransforms.ToTensor(args.norm_value),
        SPtransforms.Normalize(mean=mean, std=std)
    ])

    # TODO: Add variable downsample factor depending on the number of frames in a video
    # The idea is that a video with an higher frame rate should have an higher downsample factor in order to span
    # a longer temporal window.
    train_temporal_transform = TPtransforms.TemporalRandomCrop(args.sample_duration, args.downsample)

    # Initialize spatial and temporal transforms (validation versions)
    val_spatial_transform = SPtransforms.Compose([
        SPtransforms.Scale(args.sample_size),
        SPtransforms.CenterCrop(args.sample_size),
        SPtransforms.ToTensor(args.norm_value),
        SPtransforms.Normalize(mean=mean, std=std)
    ])

    val_temporal_transform = TPtransforms.TemporalCenterCrop(args.sample_duration, args.downsample)

    # Initialize spatial and temporal transforms (test versions)
    test_spatial_transform = SPtransforms.Compose([
        SPtransforms.Scale(args.sample_size),
        SPtransforms.CornerCrop(args.sample_size, crop_position='c'), # Central Crop in Test
        SPtransforms.ToTensor(args.norm_value),
        SPtransforms.Normalize(mean=mean, std=std)
    ])

    test_temporal_transform = TPtransforms.TemporalRandomCrop(args.sample_duration, args.downsample)

    # Load Train/Val/Test SignalForHelp Datasets
    train_dataset = Signal4HelpDataset(os.path.join(args.annotation_path, 'train_annotations.txt'), 
                                 spatial_transform=train_spatial_transform,
                                 temporal_transform=train_temporal_transform)
    
    val_dataset = Signal4HelpDataset(os.path.join(args.annotation_path, 'val_annotations.txt'), 
                                 spatial_transform=val_spatial_transform,
                                 temporal_transform=val_temporal_transform)
    
    test_dataset = Signal4HelpDataset(os.path.join(args.annotation_path, 'test_annotations.txt'), 
                                spatial_transform=test_spatial_transform,
                                temporal_transform=test_temporal_transform)
    
    # partial_collate_fn = functools.partial(collate_fn, transform=video_transforms)

    print('Size of Train Set: {}'.format(len(train_dataset)))
    print('Size of Validation Set: {}'.format(len(val_dataset)))
    print('Size of Test Set: {}'.format(len(test_dataset)))

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # Initialize DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    if args.pretrained_path == 'auto':
        # 'Build' path for pretrained weights with provided information
        if args.model in ['mobilenet', 'mobilenetv2']:
            base_model_path='models/pretrained/jester/jester_{model}_1.0x_RGB_16_best.pth'.format(model=args.model)
        else:
            base_model_path='models/pretrained/jester/jester_squeezenet_RGB_16_best.pth'
    else:
        # User provided entire path for pre-trained weights
        base_model_path = args.pretrained_path 

    model = build_model(model_path=base_model_path, 
                        type=args.model, 
                        gpus=list(range(0, num_gpus)),
                        sample_size=args.sample_size,
                        sample_duration=args.sample_duration,
                        finetune=True)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)

    if args.nesterov:
        args.dampening = 0.

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

    criterion = nn.CrossEntropyLoss()

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
          device=device, 
          pbar=pbar)

    # Load the best checkpoint obtained until now
    best_checkpoint=torch.load(os.path.join(args.model_save_path, f'best_model_{args.exp}.h5'))
    model.load_state_dict(best_checkpoint)
  
    test(loader=test_dataloader, 
         model=model,
         criterion=criterion,
         device=device)
    
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
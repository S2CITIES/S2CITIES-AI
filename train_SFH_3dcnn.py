import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorchvideo.transforms import UniformTemporalSubsample, Normalize
from torch.utils.data import DataLoader, random_split
from data.SFHDataset.dataset import Signal4HelpDataset
from build_models import build_model
import numpy as np
import functools
from tqdm import tqdm

import argparse

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(
    prog = 'Training Script for 3D-CNN models on SFH Dataset'
)
parser.add_argument('--exp', help='Name of the experiment', type=str, dest='exp', default='training_exp')
parser.add_argument('--epochs', help='Number of training epochs', type=int, dest='epochs', default=100)
parser.add_argument('--batch', help='Batch size for training with minibatch SGD', type=int, dest='batch', default=32)
parser.add_argument('--optimizer', help='Optimizer for Model Training', type=str, choices=['SGD', 'Adam'], default='SGD')
args = parser.parse_args()

writer = SummaryWriter(f'./experiments/{args.exp}')

# "Collate" function for our dataloaders
def collate_fn(batch, transform):
    # NOTE (IMPORTANT): Normalize (pytorchvideo.transforms) from PyTorchVideo wants a volume with shape CTHW, 
    # which is then internally converted to TCHW, processed and then again converted to CTHW.
    # Because our volumes are in the shape TCHW, we convert them to CTHW here, instead of doing it inside the training loop.

    videos = [transform(video.permute(1, 0, 2, 3)) for video, _ in batch]
    labels = [label for _, label in batch]

    videos = torch.stack(videos)
    labels = torch.tensor(labels)
    return videos, labels

def train(model, optimizer, scheduler, criterion, train_loader, val_loader, val_step, num_epochs, device, pbar=None):
    
    best_val_accuracy = 0

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

            optimizer.zero_grad()

            logits = model(videos)

            labels = labels.to(device)

            loss = criterion(logits, labels)
            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if pbar:
                pbar.update(videos.shape[0])

            y_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            corrects += (y_preds == labels).sum().item()
            totals += y_preds.shape[0]

        torch.save(model.state_dict(), f'models/saves/model_{args.exp}_epoch_{epoch}.h5')

        avg_train_loss = np.array(epoch_loss).mean()
        train_accuracy = 100 * corrects / totals
        print("[Epoch {}] Avg Loss: {}".format(epoch, avg_train_loss))
        print("[Epoch {}] Train Accuracy {:.2f}%".format(epoch, train_accuracy))
        
        writer.add_scalars(main_tag='train_accuracy', tag_scalar_dict={
                    'accuracy': train_accuracy,
                }, global_step=epoch)
        writer.add_scalars(main_tag='avg_train_loss', tag_scalar_dict={
                    'loss': avg_train_loss,
                }, global_step=epoch)

        # Validate/Test (if no val/dev set) the model every <validation_step> epochs of training
        if epoch % val_step == 0:
            # NOTE: test function validates the model, when it takes in input the loader for the validation set
            val_accuracy, val_loss = test(loader=val_loader, model=model, criterion=criterion, device=device, epoch=epoch)
            scheduler.step(val_loss)
            if val_accuracy > best_val_accuracy:
                # Save the best model based on validation accuracy metric
                torch.save(model.state_dict(), f'models/saves/best_model_{args.exp}.h5')
                best_val_accuracy = val_accuracy


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
        # Save metrics with tensorboard
        writer.add_scalars(main_tag='val_accuracy', tag_scalar_dict={
            'accuracy': val_accuracy
        }, global_step=epoch)
        writer.add_scalars(main_tag='val_loss', tag_scalar_dict={
            'loss': val_loss
        }, global_step=epoch)

        print('[Epoch {}] Validation Accuracy: {:.2f}%'.format(epoch, val_accuracy))
    else:
        print('Test Accuracy: {:.2f}%'.format(val_accuracy))

    return val_accuracy, val_loss

if __name__ == '__main__':

    batch_size=args.batch
    num_epochs=args.epochs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device {}".format(device))

    video_path = "./dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_ratio1_224x224"

    # Create the VideoDataset and DataLoader
    dataset = Signal4HelpDataset(video_path, 
                                 image_width=224, 
                                 image_height=224,
                                 dataset_source='dataset_noBB_224x224.pkl',
                                 preprocessing_on=False,
                                 load_on_demand=True,
                                 extract_bb_region=False,
                                 resize_frames=False)

    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])

    video, label = train_dataset[0] 
    print(video.shape)
    T, C, H, W = video.shape
    print(f"Video shape = {(T, C, H, W)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Testing if it works correctly
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    # Compute mean and std on the training set
    # Accumulate the sum and squared sum for each channel
    n_samples = 0
    channel_sum = 0
    channel_squared_sum = 0

    for data in train_dataloader:  # Iterate over the dataset or dataloader
        videos, _ = data  # Assuming images are the input data and _ represents the labels/targets
        batch_size = videos.size(0)
        channel_sum += torch.sum(videos, dim=(0,1,3,4)) 
        channel_squared_sum += torch.sum(videos ** 2, dim=(0,1,3,4))
        n_samples += batch_size

    # Compute the mean and std values for each channel
    mean = channel_sum / (n_samples*T*H*W)
    std = torch.sqrt((channel_squared_sum / (n_samples*T*H*W)) - (mean ** 2))

    # Note that the ToTensor and TemporalRandomCrop Transformations are already applied inside the Dataset class.
    video_transforms = transforms.Compose([
        Normalize(mean=mean, std=std) # Normalize from Pytorchvideo (not torchvision.transforms)
    ])
    
    partial_collate_fn = functools.partial(collate_fn, transform=video_transforms)

    # Create again DataLoader for training set
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=partial_collate_fn)
    # And other DataLoaders
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=partial_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=partial_collate_fn)

    print('Size of Train Set: {}'.format(len(train_dataset)))
    print('Size of Validation Set: {}'.format(len(val_dataset)))
    print('Size of Test Set: {}'.format(len(test_dataset)))

    print(f"Mean of the Training Set: {mean}")
    print(f"Std. of the Training Set: {std}")

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    model = build_model(base_model_path='models/pretrained/jester/jester_mobilenet_1.0x_RGB_16_best.pth', 
                        type='mobilenet', 
                        gpus=list(range(0, num_gpus)))
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)

    classifier = model.module.get_submodule('classifier')
    # optimizer = torch.optim.SGD(list(classifier.parameters()), 
    #                                             lr=0.1, 
    #                                             momentum=0.9, 
    #                                             weight_decay=0.01,
    #                                             nesterov=True)
    
    optimizer = torch.optim.Adam(list(classifier.parameters()))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min')

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(total=len(train_dataset))

    train(model=model, 
          optimizer=optimizer,
          scheduler=scheduler,
          criterion=criterion, 
          train_loader=train_dataloader,
          val_loader=val_dataloader,
          val_step=1,  
          num_epochs=num_epochs, 
          device=device, 
          pbar=pbar)
    
    test(loader=test_dataloader, 
         model=model,
         criterion=criterion,
         device=device)
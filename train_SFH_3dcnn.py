import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorchvideo.transforms import UniformTemporalSubsample
from torch.utils.data import DataLoader, random_split
from dataset.SFHDataset.dataset import Signal4HelpDataset
from build_models import build_model
import numpy as np
from tqdm import tqdm

import argparse

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(
    prog = 'Training Script for 3D-CNN models on SFH Dataset'
)
parser.add_argument('--exp', help='Name of the experiment', type=str, dest='exp', default='training_exp')
parser.add_argument('--epochs', help='Number of training epochs', type=int, dest='epochs', default=100)
parser.add_argument('--batch', help='Batch size for training with minibatch SGD', type=int, dest='batch', default=32)
args = parser.parse_args()

writer = SummaryWriter(f'./experiments/{args.exp}')



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
            videos = videos.permute(0, 2, 1, 3, 4) # (B, T, C, H, W) -> (B, C, T, H, W) 
            videos = videos.float()
            videos = videos.to(device) # Send inputs to CUDA

            optimizer.zero_grad()
            logits = model(videos)

            labels = labels.to(device)

            loss = criterion(logits, labels)
            loss.backward()
            epoch_loss.append(loss.item())
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
            videos = videos.permute(0, 2, 1, 3, 4) # (B, T, C, H, W) -> (B, C, T, H, W)
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

    batch_size=32
    num_epochs=100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device {}".format(device))

    video_path = "./dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_ratio1_224x224"

    # Define any transforms you want to apply to the videos
    transform = transforms.Compose([
        UniformTemporalSubsample(num_samples=16, temporal_dim=0)
        # Avoid normalization for now - It will be done after computing mean and std for the training set
    ])

    # Create the VideoDataset and DataLoader
    dataset = Signal4HelpDataset(video_path, 
                                 image_width=112, 
                                 image_height=112,
                                 dataset_source='dataset_noBB_224x224.pkl',
                                 preprocessing_on=False,
                                 load_on_demand=True,
                                 extract_bb_region=False,
                                 resize_frames=False, 
                                 transform=transform)

    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])

    # train_videos = [video for video, _ in train_dataset]
    # train_videos = torch.stack(train_videos) #(N, T, C, H, W)
    # trainset_mean = torch.mean(train_videos, dim=(0, 1, 3, 4))
    # trainset_std = torch.mean(train_videos, dim=(0, 1, 3, 4))

    # print(trainset_mean)
    # print(trainset_std)

    # dataset.set_mean_std()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Testing if it works correctly
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    print('Size of Train Set: {}'.format(len(train_dataset)))
    print('Size of Validation Set: {}'.format(len(val_dataset)))
    print('Size of Test Set: {}'.format(len(test_dataset)))

    video, label = train_dataset[0] 
    print(video.shape)
    T, C, H, W = video.shape
    print(f"Video shape = {(T, C, H, W)}")

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    model = build_model(base_model_path='models/pretrained/jester/jester_mobilenet_1.0x_RGB_16_best.pth', 
                        type='mobilenet', 
                        gpus=list(range(0, num_gpus)))
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)

    classifier = model.module.get_submodule('classifier')
    optimizer = torch.optim.SGD(list(classifier.parameters()), lr=0.1, dampening=0.9, weight_decay=0.01, nesterov=True)
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
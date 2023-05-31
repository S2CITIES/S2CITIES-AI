import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset.NVGesture.loader import NVGestureColorDataset 
from models.c3d import C3D
import numpy as np
from tqdm import tqdm

def train(model, optimizer, criterion, train_loader, val_loader, val_step, num_epochs, device, pbar=None):
    
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
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

            if pbar:
                pbar.update(videos.shape[0])

            y_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            corrects += (y_preds == labels).sum().item()
            totals += y_preds.shape[0]

        torch.save(model.state_dict(), 'models/saves/c3d_current.h5')
        # Validate the model every <validation_step> epochs of training
        print("[Epoch {}] Avg Loss: {}".format(epoch, np.array(epoch_loss).mean()))
        print("[Epoch {}] Train Accuracy {:.2f}%".format(epoch, 100 * corrects / totals))
        if epoch % val_step == 0:
            val_accuracy = test(loader=val_loader, model=model, device=device, epoch=epoch)
            if val_accuracy > best_val_accuracy:
                # Save the best model based on validation accuracy metric
                torch.save(model.state_dict(), 'models/saves/c3d_best.h5')
                best_val_accuracy = val_accuracy
     

def test(loader, model, device, epoch=None):
    totals = 0
    corrects = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(loader):
            videos, labels = data
            videos = videos.float()
            videos = videos.to(device)
            logits = model(videos)

            y_true.append(labels)

            y_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            y_preds = y_preds.detach().cpu()
            y_pred.append(y_preds)

            corrects += (y_preds == labels).sum().item()
            totals += y_preds.shape[0]

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    test_accuracy = 100 * corrects / totals
    if epoch is not None:
        print('[Epoch {}] Validation Accuracy: {:.2f}%'.format(epoch, test_accuracy))
    else:
        print('Test Accuracy: {:.2f}%'.format(test_accuracy))

    return test_accuracy

if __name__ == '__main__':
    torch.manual_seed(42) # Reproducibility Purposes

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((60, 80), antialias=True)
    ])

    batch_size=4
    num_epochs=100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device {}".format(device))

    train_dataset = NVGestureColorDataset(annotations_file='dataset/NVGesture/nvgesture_train_correct_cvpr2016.lst', 
                                          path_prefix='dataset/NVGesture',
                                          transforms=transform)
    test_dataset = NVGestureColorDataset(annotations_file='dataset/NVGesture/nvgesture_test_correct_cvpr2016.lst',
                                         path_prefix='dataset/NVGesture',
                                         transforms=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Testing if it works correctly
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    print('Size of Train Set: {}'.format(len(train_dataset)))
    print('Size of Test Set: {}'.format(len(test_dataset)))

    video, label = train_dataset[0]
    C, T, H, W = video.shape

    model = C3D(channels=C, length=T, height=H, width=W, tempdepth=3, outputs=25)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(list(model.parameters()))
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    step = 1
    model_parameters = sum(p.numel() for p in model.parameters())

    print('Total number of parameters: {}'.format(model_parameters))

    pbar = tqdm(total=len(train_dataset))

    train(model=model, 
          optimizer=optimizer, 
          criterion=criterion, 
          train_loader=train_dataloader,
          val_loader=test_dataloader,
          val_step=5,  
          num_epochs=num_epochs, 
          device=device, 
          pbar=pbar)
    
    test(loader=test_dataloader, 
         model=model,
         device=device)
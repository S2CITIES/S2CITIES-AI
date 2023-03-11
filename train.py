import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.c3d import C3D
import numpy as np

def collate_fn(data):
    videos, labels = [video for video, audio, label in data], \
                        [label for video, audio, label in data]
    return torch.stack(videos, dim=0).to(torch.float), torch.reshape(torch.tensor(labels), (-1, 1)).to(torch.float)

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, validation_step):
    best_val_accuracy = 0

    ############### Training ##################
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        corrects = 0
        totals = 0

        for i, data in enumerate(train_loader):
            videos, labels = data
            videos = videos.permute(0, 2, 1, 3, 4) # From BTCHW to BCTHW 
            videos = videos.to(device) # Send inputs to CUDA

            optimizer.zero_grad()
            logits = model(videos)

            labels = labels.to(device)

            loss = criterion(logits, labels)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

            y_preds = torch.argmax(torch.softmax(logits), dim=1)
            corrects += (y_preds == labels).sum().item()
            totals += y_preds.shape[0]

        if epoch % validation_step == 0:
            # Validate the model every <validation_step> epochs of training
            print("[Epoch {}] Avg Loss: {}".format(epoch, np.array(epoch_loss).mean()))
            print("[Epoch {}] Train Accuracy {:.2f}%".format(epoch, 100 * corrects / totals))
            val_accuracy = test(val_loader, model, epoch)
            if val_accuracy > best_val_accuracy:
                # Save the best model based on validation accuracy metric
                torch.save(model.state_dict(), 'models/saves/clp.model')
                best_val_accuracy = val_accuracy
     

def test(loader, model, epoch=None):
    totals = 0
    corrects = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(loader):
            videos, labels = data
            videos = videos.permute(1, 0, 2, 3) # From TCHW to CTHW 
            logits = model(videos)

            y_true.append(labels)

            y_preds = torch.argmax(torch.softmax(logits), dim=1)
            y_preds = y_preds.detach().cpu()
            y_pred.append(y_preds)

            corrects += (y_preds == labels).sum().item()
            totals += y_preds.shape[0]

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    test_accuracy = 100 * corrects / totals
    if epoch is not None:
        print('[Epoch {}] Test Accuracy: {:.2f}%'.format(epoch, test_accuracy))
    else:
        print('Test Accuracy: {:.2f}%'.format(test_accuracy))

    return test_accuracy

if __name__ == '__main__':
    torch.manual_seed(42) # Reproducibility Purposes

    dataset = torchvision.datasets.UCF101(root='dataset/UCF101/data', 
                                          annotation_path='dataset/UCF101/labels',
                                          frames_per_clip=16,
                                          output_format='TCHW')
    
    batch_size=32
    num_epochs=100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device {}".format(device))

    train_dataset, test_dataset, val_dataset = random_split(dataset, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Size of Train Set: {}'.format(len(train_dataset)))
    print('Size of Validation Set: {}'.format(len(val_dataset)))
    print('Size of Test Set: {}'.format(len(test_dataset)))

    video, _, label = dataset[0]
    T, C, H, W = video.shape

    model = C3D(channels=C, length=T, height=H, width=W, tempdepth=3, outputs=101)
    optimizer = torch.optim.Adam(list(model.parameters()))
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    step = 1
    model_parameters = sum(p.numel() for p in model.parameters())

    print('Total number of parameters: {}'.format(model_parameters))

    train(model=model, 
          optimizer=optimizer, 
          criterion=criterion, 
          train_loader=train_loader, 
          val_loader=val_loader, 
          num_epochs=num_epochs, 
          validation_step=1)
    test(test_loader, model)
import os
import torch
import torch.nn as nn
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
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

from matplotlib import rc
rc('animation', html='jshtml')

input = None
feat_maps = None

def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w))
        cam = np.matmul(weight[idx], beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def save_video(video):
    print("Saving video..")
    fig, ax = plt.subplots()

    video_cpu = video.cpu().numpy()

    frames = [[ax.imshow(video_cpu[i])] for i in range(len(video_cpu))]

    ani = animation.ArtistAnimation(fig, frames)
    ani.save("../gdrive/MyDrive/DRIVE S2CITIES/Artificial Intelligence/input_video2.mp4")
    

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
        save = True

        for i, data in enumerate(loader):
            videos, labels = data
            videos = videos.float()
            videos = videos.to(device)

            logits, inp, f_maps = model(videos)
            
            labels = labels.to(device)
            val_loss_batch = criterion(logits, labels)

            val_loss.append(val_loss_batch.item())

            y_true.append(labels)

            y_preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            corrects += (y_preds == labels).sum().item()
            totals += y_preds.shape[0]

            y_preds = y_preds.detach().cpu()
            y_pred.append(y_preds)

            if save and labels[0]==1:
                inp = inp.to(torch.device("cpu"))
                f_maps = f_maps.to(torch.device("cpu"))
                global input, feat_maps
                input = inp
                feat_maps = f_maps
                print(f"logits: {logits}, y_pred: {y_preds}, labels: {labels}")
                save = False


    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    val_accuracy = 100 * corrects / totals
    val_loss = np.array(val_loss).mean()

    if epoch is not None:
        print('[Epoch {}] Validation Accuracy: {:.2f}%'.format(epoch, val_accuracy))
    else:
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

    val_dataset = Signal4HelpDataset(os.path.join(args.annotation_path, 'val_annotations.txt'), 
                                 spatial_transform=val_spatial_transform,
                                 temporal_transform=val_temporal_transform)
    
    print('Size of Validation Set: {}'.format(len(val_dataset)))

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

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
    cam_model.module.checked = True

    #print(cam_model.module.classifier[1].weight.shape) torch.Size([2, 1280])

    val_accuracy, val_loss = test(loader=val_dataloader, model=cam_model, criterion=criterion, device=device, epoch=None)

    out_cams = return_CAM(feat_maps.squeeze(dim=2), cam_model.module.classifier[1].weight.cpu(), [0,1])
    print(f"out_cams len: {len(out_cams)}, out_cams[0].shape: {out_cams[0].shape}")

    #print(f"video shape before permute: {input.shape}")
    #input = input[0].permute(1,2,3,0) # Permuting to (Bx)HxWxC format
    #input = input[...,[2,1,0]]
    #print(f"video shape after permute: {input.shape}")
    #save_video(input)

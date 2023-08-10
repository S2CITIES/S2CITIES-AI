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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

colormap = plt.get_cmap('coolwarm')

input = None
feat_maps = None

def choose_scale(img, scale="grayscale"):
    if (scale == "coolwarm"):
        img = colormap(img)[:, :, :3]
        img = np.uint8(255 * img)
    else:
        img = np.uint8(255 * img)
    return img

def save_CAM(feature_conv, weight, class_idx):
    size_upsample = (args.sample_size, args.sample_size)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for sample in feature_conv:
        sample_cam = []
        for idx in class_idx:
            beforeDot =  sample.reshape((nc, h*w))
            cam = np.matmul(weight[idx], beforeDot)
            cam = cam.reshape(h, w).numpy()
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = choose_scale(cam_img, "coolwarm")
            print(f"feature_conv: :{feature_conv.shape}, sample: :{sample.shape}, beforeDot: :{beforeDot.shape}, weight[{idx}]: {weight.shape}, cam: {cam.shape}")
            sample_cam.append(cv2.resize(cam_img, size_upsample))
        output_cam.append(list(sample_cam))
    
    for i in range(len(output_cam)):
        for j in range(len(output_cam[i])):
            cv2.imwrite(f"../gdrive/MyDrive/DRIVE S2CITIES/Artificial Intelligence/CAM Analysis/cam_sample{i}_class{j}.png", output_cam[i][j])
    return output_cam

def save_video(input):
    print("Saving videos..")

    for i, inp in enumerate(input):
        video = inp.permute(1,2,3,0) # Permuting to Tx(HxWxC)
        #video = video[...,[2,1,0]]

        fig, ax = plt.subplots()

        video_cpu = video.cpu().numpy()

        for ind in range(len(video_cpu)):
            for j in range(len(video_cpu[ind])):
                video_cpu[ind][j] = (video_cpu[ind][j] * std[j]) + mean[j]

        video_cpu = np.uint8(video_cpu)

        frames = [[ax.imshow(video_cpu[i])] for i in range(len(video_cpu))]

        ani = animation.ArtistAnimation(fig, frames)
        ani.save(f"../gdrive/MyDrive/DRIVE S2CITIES/Artificial Intelligence/CAM Analysis/sample{i}.mp4")


def save_video_v2(input):

    print("Saving videos..")
    
    for i, inp in enumerate(input):
        video = inp.permute(1,2,3,0) # Permuting to Tx(HxWxC)
        video = video[...,[2,1,0]]
        video_cpu = video.cpu().numpy()
        print(video_cpu)
                
        min_per_frame = np.min(video_cpu, axis=(1, 2, 3))
        max_per_frame = np.max(video_cpu, axis=(1, 2, 3))

        for ind in range(len(video_cpu)):
            video_cpu[ind] = (video_cpu[ind] * std) + mean
            video_cpu[ind] = (video_cpu[ind] - min_per_frame[ind]) / max_per_frame[ind]

        video_cpu = np.uint8(255 * video_cpu)

        print(video_cpu)

        writer = cv2.VideoWriter(filename=f"../gdrive/MyDrive/DRIVE S2CITIES/Artificial Intelligence/CAM Analysis/v2_sample{i}.mp4",
                                 fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=6.4,
                                 frameSize=(int(args.sample_size), int(args.sample_size)), isColor=True)

        if writer.isOpened:
            for i in range(len(video_cpu)):
                writer.write(video_cpu[i])

            writer.release()
        else:
            print("Error opening the file!")

    

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

        for _, data in enumerate(loader):
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

            # If the batch size is > 1, I want both a positive and a negative
            # Otherwise I want a positive
            global input, feat_maps
            if save and labels.shape[0]!=1 and (1 in labels) and (0 in labels):
                inp = inp.to(torch.device("cpu"))
                f_maps = f_maps.to(torch.device("cpu"))
                input = inp
                feat_maps = f_maps
                print(f"logits: {logits}, y_pred: {y_preds}, labels: {labels}")
                save = False
            elif save and labels.shape[0]==1 and (1 in labels):
                inp = inp.to(torch.device("cpu"))
                f_maps = f_maps.to(torch.device("cpu"))
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

    val_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=args.num_workers)

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




    last_layer = cam_model.module.classifier[1]
    for param in last_layer.parameters():
        param.requires_grad = False
    weights = last_layer.weight.cpu()

    out_cams = save_CAM(feat_maps.squeeze(dim=2), weights, [0,1])
    save_video(input)
    save_video_v2(input)

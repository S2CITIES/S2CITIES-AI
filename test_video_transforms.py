from torchvideotransforms import video_transforms, volume_transforms
from torch.utils.data import DataLoader, random_split
from dataset.NVGesture.loader import NVGestureColorDataset 

train_dataset = NVGestureColorDataset(annotations_file='dataset/NVGesture/nvgesture_train_correct_cvpr2016.lst', 
                                        path_prefix='dataset/NVGesture',
                                        transforms=None,
                                        tensor=False)

print(train_dataset[0][0].shape)

video_transform_list = [video_transforms.Resize((120, 160)),
                        video_transforms.RandomRotation(30),
			            video_transforms.RandomCrop((112, 112)),
			            volume_transforms.ClipToTensor()]
transforms = video_transforms.Compose(video_transform_list)

print(transforms(train_dataset[0][0]).shape) # Output shape (after ClipToTensor) seems to be C, T, H, W
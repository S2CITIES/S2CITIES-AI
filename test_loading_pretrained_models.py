import torch
import torch.nn as nn
from models.mobilenetv2 import MobileNetV2
from models.mobilenet import (
    get_model as get_model_mobilenet
)

model_path = 'models/pretrained/jester/jester_mobilenet_1.0x_RGB_16_best.pth'
model = get_model_mobilenet(num_classes=27, sample_size = 224, width_mult=1.)
model = model.cuda()
model = nn.DataParallel(model, device_ids=None)

checkpoint = torch.load(model_path)

print(checkpoint['state_dict'])

model.load_state_dict(checkpoint['state_dict'])

print(model)

# Testing the model with a random input
input_var = torch.randn((8, 3, 16, 224, 224)) # (B, C, T, H, W)
output = model(input_var)
print(output.shape)
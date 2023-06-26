import torch
import torch.nn as nn
from models.mobilenetv2 import MobileNetV2
from models.mobilenet import (
    get_model as get_model_mobilenet
)

def build_model(base_model_path, type='mobilenet', gpus=None):
    # All models pretrained on Jester (27 classes)
    if type == 'mobilenet':
        model=get_model_mobilenet(num_classes=27, sample_size = 224, width_mult=1.)
        model=model.cuda()
        model=nn.DataParallel(model, device_ids=gpus)
        checkpoint=torch.load(base_model_path)
        model.load_state_dict(checkpoint['state_dict'])

        # Freeze model weights
        for param in list(model.parameters()):
            param.requires_grad = False

        classifier = model.module.get_submodule('classifier')
        print(f"Previous classifier: {classifier}")

        # Weights of the new classifier will be fine-tunable
        new_classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1024, out_features=2, bias=True)
        )
        new_classifier.cuda()
        model.module.classifier = new_classifier
        classifier = model.module.get_submodule('classifier')
        print(f"New classifier: {classifier}")
        print("New model built successfully")
        return model
    
    else:
        print("To be implemented")

if __name__ == '__main__':
    build_model(base_model_path='models/pretrained/jester/jester_mobilenet_1.0x_RGB_16_best.pth')
import torch
import torch.nn as nn
from models.mobilenetv2 import (
    get_model as get_model_mobilenetv2
)
from models.mobilenet import (
    get_model as get_model_mobilenet
)
from models.squeezenet import (
    get_model as get_model_squeezenet
)

# Recursive function to set BatchNorm layers to evaluation mode
def set_bn3d_eval_mode(layer):
    if isinstance(layer, nn.BatchNorm3d):
        layer.eval()
    for child_module in layer.children():
        print(child_module)
        set_bn3d_eval_mode(child_module)

def build_model(model_path, type='mobilenet', gpus=None, num_classes=27, sample_size=112, sample_duration=16, width_mult=1., output_features=2, finetune=True, state_dict=False):
    # All models pretrained on Jester (27 classes)
    if type == 'mobilenet':
        model=get_model_mobilenet(num_classes=num_classes, sample_size=sample_size, width_mult=width_mult)
    elif type=='mobilenetv2':
        model=get_model_mobilenetv2(num_classes=num_classes, sample_size=sample_size, width_mult=width_mult)
    elif type=='squeezenet':
        model=get_model_squeezenet(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes)
    else:
        print("Unknown model type. Select between: mobilenet, mobilenetv2, squeezenet.")
        return None
        
    model=nn.DataParallel(model, device_ids=gpus)
    model=model.cuda()

    if model_path: # If a model_path is provided, load the trained checkpoint for finetuning or resuming training.
        
        checkpoint=torch.load(model_path)
        if not state_dict: 
            # Not a state_dict, but the entire model dictionary
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Only state_dict was saved, so checkpoint is already a state_dict
            model.load_state_dict(checkpoint)

        if finetune:
            # Freeze model weights
            for param in list(model.parameters()):
                param.requires_grad = False
            # Set BatchNorm layers of the feature network to eval mode
            set_bn3d_eval_mode(model)

            classifier = model.module.get_submodule('classifier')
            print(f"Previous classifier: {classifier}")

            if type == 'squeezenet': # Version 1.0 or 1.1
                # Weights of the new classifier will be fine-tunable
                last_duration = model.module.last_duration
                last_size = model.module.last_size
                new_classifier = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Conv3d(512, output_features, kernel_size=1),
                    nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
                )
                torch.nn.init.kaiming_normal_(new_classifier[1].weight, mode='fan_out')
            else:
                new_classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=False),
                    nn.Linear(in_features=model.module.last_channel, out_features=output_features, bias=True),
                )

            new_classifier.cuda()
            model.module.classifier = new_classifier

            print(f"New classifier: {new_classifier}")
            print("New model built successfully")

    # Test the model
    x = torch.randn((2, 3, sample_duration, sample_size, sample_size))
    out = model(x)
    print(f"Testing the model - Obtained: {out.shape}, Expected: torch.Size([2, {output_features}])")

    return model

if __name__ == '__main__':
    # build_model(model_path='models/pretrained/jester/jester_mobilenet_1.0x_RGB_16_best.pth')
    # build_model(model_path='models/pretrained/jester/jester_mobilenetv2_1.0x_RGB_16_best.pth', type="mobilenetv2")
    build_model(model_path='models/pretrained/jester/jester_squeezenet_RGB_16_best.pth', type='squeezenet')
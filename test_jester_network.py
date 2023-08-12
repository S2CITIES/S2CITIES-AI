# Testing how the pre-trained model work on Jester dataset
from build_models import build_model


if __name__ == '__main__':
    base_model_path = './models/pretrained/jester_mobilenetv2_1.0x_RGB_16_best.pth'
    model = build_model(model_path=base_model_path, 
                    type='mobilenetv2', 
                    gpus=[0],
                    sample_size=112,
                    sample_duration=16,
                    num_classes=27,
                    finetune=False)
    print(model)

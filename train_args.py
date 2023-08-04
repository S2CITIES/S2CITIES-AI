import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'Training Script for 3D-CNN models on SFH Dataset'
    )
    # Supported Models
    model_choices = ['mobilenet',
                    'mobilenetv2', 
                    'squeezenet']
    
    train_crop_choices = ['corner', 'random', 'center']
    
    parser.add_argument('--exp', help='Name of the experiment', type=str, dest='exp', default='training-exp-default')
    parser.add_argument('--epochs', help='Number of training epochs', type=int, dest='epochs', default=100)
    parser.add_argument('--batch', help='Batch size for training with minibatch SGD', type=int, dest='batch', default=32)
    parser.add_argument('--optimizer', help='Optimizer for Model Training', type=str, choices=['SGD', 'Adam'], default='SGD')
    parser.add_argument('--early_stop_patience', help='Patience for early stopping criteria', type=int, default=5)
    parser.add_argument('--sample_size', default=224, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=1, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str, choices=train_crop_choices, help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    
    ### SGD algorithm parameters ###
    parser.add_argument('--lr', default=0.04, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum parameter for SGD')
    parser.add_argument('--dampening', default=0.9, type=float, help='Dampening paremeter for SGD')
    parser.add_argument('--wd', default=1e-3, type=float, help='Weight Decay parameter for SGD')
    parser.add_argument('--nesterov', action='store_true', help='If true, use nesterov momentum in SGD.')
    parser.set_defaults(nesterov=False)

    ### lr scheduler parameters ### 
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 200, 250], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    
    parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--no_norm', action='store_true', help='If true, do not normalize input data.')
    parser.set_defaults(no_norm=False)
    parser.add_argument('--model', help='Select 3D-CNN model type', type=str, choices=model_choices, default='mobilenet')
    parser.add_argument('--groups', default=3, type=int, help='The number of groups at group convolutions at conv layers')
    parser.add_argument('--width_mult', default=1.0, type=float, help='The applied width multiplier to scale number of filters')
    parser.add_argument('--data_path', default='dataset/SFHDataset/SFH/SFH_Dataset_S2CITIES_simplified_ratio1_224x224', type=str, help='Path for train/test/val video files.')
    parser.add_argument('--annotation_path', default='data/SFHDataset', type=str, help='Path for train/test/val annotation file')
    parser.add_argument('--pretrained_path', help='Absolute/Relative path for pretrained weights', type=str, dest='pretrained_path', default='auto')
    parser.add_argument('--model_save_path', help='Absolute/Relative path for saving trained weights', type=str, dest='model_save_path', default='./models/saves')
    ### Was used with tensorboard
    parser.add_argument('--exp_path', help='Absolute/Relative path for saving experiment logs', type=str, dest='exp_path', default='./experiments')
    ### 
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--recompute_mean_std', action='store_true', help='If true, compute from scratch mean and std.')
    parser.set_defaults(recompute_mean_std=False)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of working threads for loaders')

    args = parser.parse_args()
    return args


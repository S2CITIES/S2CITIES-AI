import torch
import torch.nn as nn


'''
Implementation of a 3D ConvNet from the paper https://arxiv.org/pdf/1412.0767.pdf.
In a 3D ConvNet, convolutions are done spatio-temporally and not only spatially.
Like in the paper, dimensions of a video are described as c x l x h x w, where
c is the number of channels
l is the length of a video, in number of frames
h and w are the height and the width of a single frame, respectively

When referring to a spatio-temporal kernel, with shape d x k x k, 
d is the kernel temporal depth
k is the spatial size
'''
class C3D(nn.Module):
    def __init__(self, channels, length, height, width, tempdepth):
        super().__init__()
        self.c = channels
        self.l = length
        self.h = height
        self.w = width
        self.d = tempdepth
        self.conv1 = nn.Conv3d(in_channels=self.c, out_channels=64, kernel_size=(self.d, 3, 3), padding=1, stride=1)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(self.d, 3, 3), padding=1, stride=1)
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(self.d, 3, 3), padding=1, stride=1)
        self.conv4 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(self.d, 3, 3), padding=1, stride=1)
        self.conv5 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(self.d, 3, 3), padding=1, stride=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2))
        self.fc1 = nn.Linear(in_features=256*(self.l//16)*(self.h//32)*(self.w//32), out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # input shape (N, c, l, h, w)
        out1 = self.pool1(self.conv1(x))
        # out1 shape (N, 64, l, h//2, w//2)
        out2 = self.pool(self.conv2(out1))
        # out2 shape (N, 128, l//2, h//4, w//4)
        out3 = self.pool(self.conv3(out2))
        # out3 shape (N, 256, l//4, h//8, w//8)
        out4 = self.pool(self.conv4(out3))
        # out4 shape (N, 256, l//8, h//16, w//16)
        out5 = self.pool(self.conv4(out4))
        # out5 shape (N, 256, l//16, h//32, w//32) -> should be (N, 256, 1, 8, 8) in the current setup

        if len(x.shape) == 5:
            # Batched input, consider this when flattening
            out = torch.flatten(out5, start_dim=1)
        else:
            out = torch.flatten(out5)

        lin = torch.relu(self.fc1(out)) # Output of the first linear layer, after ReLU activation
        return torch.sigmoid(self.fc2(lin)) # Output of the second linear layer (single output) after Sigmoid activation

        
if __name__ == '__main__':
    # Testing the network
    # Simulating an input with batch size = 2, channels = 3 (RGB Video) and sequence length = 16 (number of frames)
    # Every video is reshaped with a frame shape (256, 256)
    x = torch.randn((2, 3, 16, 256, 256))
    model = C3D(channels=3, length=16, height=256, width=256, tempdepth=3)
    out = model(x)
    print(out.shape) # Expected torch.Size([2, 1])

    # Test also with unbatched input
    x = torch.randn((3, 16, 256, 256))
    out = model(x)
    print(out.shape) # Expected torch.Size([1])
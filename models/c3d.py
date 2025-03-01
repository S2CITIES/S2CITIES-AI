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
    def __init__(self, channels, length, height, width, tempdepth, outputs):
        super().__init__()
        self.c = channels   # number of channels per frame (e.g., RGB = 3 channels)
        self.l = length     # number of frames per input video
        self.h = height
        self.w = width
        self.d = tempdepth  # temporal depth of kernels 
        self.conv1 = nn.Conv3d(in_channels=self.c, out_channels=32, kernel_size=(1, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=32)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=64)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3), padding=1)
        self.bn345 = nn.BatchNorm3d(num_features=128)
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(self.d, 3, 3), padding=1)
        self.conv5 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(self.d, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2))
        self.fc = nn.Linear(in_features=128*(self.l//16)*(self.h//32)*(self.w//32), out_features=outputs)
        self.dropout = nn.Dropout3d(p=0.25)

    def forward(self, x):
        # input shape (N, c, l, h, w)
        out1 = self.dropout(torch.relu(self.pool1(self.bn1(self.conv1(x)))))
        # TODO: Test skip-connections with this architecture
        # print(out1.shape)
        # out1 shape (N, 64, l, h//2, w//2)
        out2 = self.dropout(torch.relu(self.pool(self.bn2(self.conv2(out1)))))
        # print(out2.shape)
        # out2 shape (N, 128, l//2, h//4, w//4)
        out3 = self.dropout(torch.relu(self.pool(self.bn345(self.conv3(out2)))))
        # print(out3.shape)
        # out3 shape (N, 256, l//4, h//8, w//8)
        out4 = self.dropout(torch.relu(self.pool(self.bn345(self.conv4(out3)))))
        # print(out4.shape)
        # out4 shape (N, 256, l//8, h//16, w//16)
        out5 = self.dropout(torch.relu(self.pool(self.bn345(self.conv5(out4)))))
        # print(out5.shape)
        # out5 shape (N, 256, l//16, h//32, w//32)

        if len(x.shape) == 5:
            # Batched input, consider this when flattening
            out = torch.flatten(out5, start_dim=1)
        else:
            out = torch.flatten(out5)

        # print(out.shape)
        return self.fc(out) # Output of linear layer before Softmax activation. The computation of the pdf (softmax) is embedded in PyTorch (nn.CrossEntropyLoss) 

        
if __name__ == '__main__':
    # Testing the network
    # Simulating an input with batch size = 2, channels = 3 (RGB Video) and sequence length = 80 (number of frames)
    # Every video is reshaped with a frame shape (60, 80) (initial resolution is downscaled by a factor of 4)
    x = torch.randn((2, 3, 16, 112, 112))
    model = C3D(channels=3, length=16, height=112, width=112, tempdepth=3, outputs=25)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    out = model(x)
    print(out.shape) # Expected torch.Size([2, 25])
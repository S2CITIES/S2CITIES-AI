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
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        return logits

class FineTunedC3D(nn.Module):

    def __init__(self, pretrained_model_path, outputs):
        super(FineTunedC3D, self).__init__()
        self.c3d = C3D()
        self.c3d.load_state_dict(torch.load(pretrained_model_path))

        # Freeze the weights of the pre-trained model.
        for param in self.c3d.parameters():
            param.requires_grad = False

        # Fine-tuning the weights of the final MLP for predictions.
        for param in self.c3d.fc6.parameters():
            param.requires_grad = True
        for param in self.c3d.fc7.parameters():
            param.requires_grad = True

        # Substituting the last "logits" layer with a new FC layer having 25 output neurons.
        self.c3d.fc8 = nn.Linear(in_features=4096, out_features=outputs)
        for param in self.c3d.fc8.parameters():
            param.requires_grad = True

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.c3d(x)

if __name__ == '__main__':
    # Testing the pre-trained network (on Sport-1M)
    x = torch.randn((2, 3, 16, 112, 112)).cuda()
    model = C3D()
    model.load_state_dict(torch.load('./pretrained/c3d.pickle'))
    model.cuda()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    out = model(x)
    print(out.shape) # Expected torch.Size([2, 487])

    finetuned_model = FineTunedC3D(pretrained_model_path='./pretrained/c3d.pickle',
                                   outputs=25)
    finetuned_model.cuda()
    out = finetuned_model(x)
    print(out.shape) # Expected torch.Size([2, 25])
                                   
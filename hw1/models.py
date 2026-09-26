import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.relu6 = nn.ReLU(inplace=True)

        self.head_gap = nn.AdaptiveAvgPool2d((1, 1)) # аналог GlobalAvgPool в PyTorch
        self.linear1 = nn.Linear(512, 256, bias=False)
        self.relu_head = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256, 100, bias=False)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.head_gap(x)
        x = torch.flatten(x, 1)
        x = self.relu_head(self.linear1(x))
        x = self.linear2(x)
        return x

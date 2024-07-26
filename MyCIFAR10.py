from collections import OrderedDict

import torch
import torch.nn as nn


class MyCIFAR10(nn.Module):
    def __init__(self):
        super(MyCIFAR10, self).__init__()
        # 给每一层命名
        self.layer1 = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)),
                ("relu1", nn.ReLU()),
                ("maxpool1", nn.MaxPool2d(kernel_size=2))
            ])
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=64*4*4, out_features=64),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_flatten = torch.flatten(x_layer3, 1)
        x_layer4 = self.layer4(x_flatten)
        y = self.layer5(x_layer4)
        return y
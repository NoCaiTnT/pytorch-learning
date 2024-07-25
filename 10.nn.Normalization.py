import torch
import torch.nn as nn

input = torch.rand(2, 2, 2, 2)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x):
        return self.bn(x)

net = MyNet()
output = net(input)
print(input)
print(output)
import torch.nn as nn
import torch
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

    def forward(self, x):
        return x+1

net = MyNet()
x = torch.tensor(1.0)
print(net(x))
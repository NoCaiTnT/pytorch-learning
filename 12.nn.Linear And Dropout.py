import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

for i, data in enumerate(dataloader):
    imgs, target = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    print(output.shape)
    break

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.linear1 = nn.Linear(64*3*32*32, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.linear1(x)
        y = self.dropout(x)
        return x, y

net = MyNet()
for i, data in enumerate(dataloader):
    imgs, target = data
    # imgs = torch.reshape(imgs, (1, 1, 1, -1))
    imgs = torch.flatten(imgs)      # 平铺, 并将维度大小为1的维度去掉
    print(imgs.shape)
    output, dropout = net(imgs)
    print(output.shape, output)
    print(dropout.shape, dropout)
    break
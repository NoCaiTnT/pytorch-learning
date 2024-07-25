import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 使用torch.nn.functional实现
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = input.unsqueeze(0).unsqueeze(0)
kernel = kernel.unsqueeze(0).unsqueeze(0)

output = F.conv2d(input, kernel, stride=1, padding=0)
print(output)

output = F.conv2d(input, kernel, stride=1, padding=1)
print(output)

# 使用torch.nn实现
dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

net = MyNet()
print(net)
writer = SummaryWriter('logs')
for i, data in enumerate(dataloader):
    imgs, target = data
    print(imgs.shape)   # torch.Size([64, 3, 32, 32])
    output = net(imgs)
    print(output.shape) # torch.Size([64, 6, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('input', imgs, i)
    writer.add_images('output', output, i)
    break
writer.close()
import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = input.unsqueeze(0).unsqueeze(0)

class MyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # ceil_mode: 向上取整, 即输入图像长度不够时, 也会进行池化
        self.maxpool = nn.MaxPool2d(3, ceil_mode=False)

    def forward(self, x):
        x = self.maxpool(x)
        return x

net = MyNet()
output = net(input)
print(output)

dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


writer = SummaryWriter('logs')
for i, data in enumerate(dataloader):
    imgs, target = data
    output = net(imgs)
    print(output.shape)
    writer.add_images('input', imgs, i)
    writer.add_images('output', output, i)
    break
writer.close()
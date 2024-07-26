import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

# 数据集
dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class MyCIFAR10(nn.Module):
    def __init__(self):
        super(MyCIFAR10, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
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
            nn.Linear(in_features=64, out_features=10),
            nn.ReLU()
        )

    def forward(self, x):
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_flatten = self.flatten(x_layer3)
        x_layer4 = self.layer4(x_flatten)
        y = self.layer5(x_layer4)
        return y

net = MyCIFAR10()
loss_fn = nn.CrossEntropyLoss()
optim = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        imgs, target = data
        output = net(imgs)
        l = loss_fn(output, target)
        optim.zero_grad()   # 梯度清零
        l.backward()        # 反向传播, 计算梯度
        optim.step()        # 更新参数
        running_loss += l.item()
    print(f'epoch: {epoch}, loss: {running_loss}')
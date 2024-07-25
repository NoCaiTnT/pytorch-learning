import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (1, 1, 2, 2))

relu = nn.ReLU()

output = relu(input)
print(output)

# ReLU的参数:
#   inplace: 是否原地操作, 默认为False
#       True: 修改输入的值
#           Input = -1
#           ReLU(Input)
#           Input = 0
#       False: 不修改输入的值
#           Input = -1
#           Output = ReLU(Input)
#           Input = -1
#           Output = 0

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)

net = MyNet()
output = net(input)
print(output)

dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter('logs')
for i, data in enumerate(dataloader):
    imgs, target = data
    output = net(imgs)
    writer.add_images('input', imgs, i)
    writer.add_images('output', output, i)
    break
writer.close()
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MyCIFAR10 import MyCIFAR10

# 0. 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载数据集
train_data = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
print(f'训练集的大小为: {len(train_data)}')
print(f'测试集的大小为: {len(test_data)}')

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 2. 定义网络模型
net = MyCIFAR10()
# net.load_state_dict(torch.load('./models/cifar10_5.pth'))
net = net.to(device)

# 3. 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 4. 定义优化器
m_optim = optim.SGD(net.parameters(), lr=0.01)

# 5. 定义scheduler
scheduler = optim.lr_scheduler.StepLR(m_optim, step_size=10, gamma=0.1)

# 6. 定义训练步数
epochs = 10

# 7. 使用tensorboard
writer = SummaryWriter('logs')

# 8. 计算准确率
def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = torch.eq(pred, target).sum().float().item()
        acc = correct / len(target)
        return acc

# 9. 训练网络
for epoch in range(epochs):
    # 训练
    net.train()     # 只对Dropout, BatchNorm等有影响
    print(f'-------------第{epoch+1}个epoch开始训练-------------')
    train_loss = 0
    for data in train_loader:
        imgs, target = data
        imgs, target = imgs.to(device), target.to(device)
        output = net(imgs)
        loss = loss_fn(output, target)

        m_optim.zero_grad()
        loss.backward()
        m_optim.step()

        train_loss += loss.item()

    print(f' epoch: {epoch+1}, loss: {train_loss / len(train_loader)}')
    scheduler.step()

    # 测试
    net.eval()
    with torch.no_grad():
        test_loss = 0
        test_acc = 0
        for j, data in enumerate(test_loader):
            imgs, target = data
            imgs, target = imgs.to(device), target.to(device)
            output = net(imgs)
            loss = loss_fn(output, target)
            test_loss += loss.item()

            test_acc += accuracy(output, target)

        print(f' epoch: {epoch+1}, test loss: {test_loss / len(test_loader)}, acc: {test_acc / len(test_loader)}')


    # 画图
    writer.add_scalars("train and test",
                      {"train_loss": train_loss / len(train_loader),
                       "test_loss": test_loss / len(test_loader),
                       "test_acc": test_acc / len(test_loader)},
                      epoch+1)

    # 保存模型
    if epoch != 0 and (epoch+1) % 5 == 0:
        torch.save(net.state_dict(), f'./models/cifar10_{epoch+1}.pth')

writer.close()
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=False)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('logs')
for epoch in range(2):
    for i, data in enumerate(test_loader):
        imgs, target = data
        # print(imgs.shape)
        # print(target.shape)
        writer.add_images(f'log {epoch}', imgs, i)
writer.close()
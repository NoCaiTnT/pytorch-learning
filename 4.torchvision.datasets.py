import torchvision
from torch.utils.tensorboard import SummaryWriter

# 下载数据集
# train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root='data', train=False, download=True)
#
# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# 对数据集进行变换
dataset_transforms = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(32, padding=4),
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=dataset_transforms)
test_set = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=dataset_transforms)

# print(test_set[0])

# 使用tensorborad显示
writer = SummaryWriter('logs')
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
import torch
import torch.nn as nn
import torchvision

vgg16 = torchvision.models.vgg16(weights=None)

# 1. 保存网络结构和参数
torch.save(vgg16, 'models/vgg16.pth')

# 1. 加载网络结构和参数
#   注意: 需要import这个模型
net = torch.load('models/vgg16.pth')
print(net)

# 2. 保存网络参数
torch.save(vgg16.state_dict(), 'models/vgg16_state_dict.pth')

# 2. 加载网络参数
net = torchvision.models.vgg16(weights=None)
net.load_state_dict(torch.load('models/vgg16_state_dict.pth'))
print(net)

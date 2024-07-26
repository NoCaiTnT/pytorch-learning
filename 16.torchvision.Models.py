import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.models as models

vgg16_true = models.vgg16(pretrained=True)
vgg16_false = models.vgg16(pretrained=False)

print(vgg16_true)

# 添加网络模型
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 修改网络模型
vgg16_true.classifier[6] = nn.Linear(4096, 10)
print(vgg16_true)
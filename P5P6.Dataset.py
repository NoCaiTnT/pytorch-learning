# 数据
#   包含所有数据
#       有用的数据
#       没用的数据

# Dataset: 提供一种方式取获取数据及其label
#   从数据中提取有用的数据并进行编号
#   获得数据对应的label
#   需要实现:
#       1. 如何获取每一个数据及其label
#       2. 告诉我们总共有多少数据

# Dataloader: 为后面的网络提供不同的数据形式
#   对数据进行打包

#   数据        Dataset           Dataloader
# 红 黄 蓝     label1 蓝 0          [0,1,2,3]
# 蓝 红 红     label2 蓝 1          [4,5,6,7]
# 黄 蓝 蓝     label3 蓝 2          [8,9,10,11]
#    ↑             ↑                   ↑
# 杂乱的数据  提取数据和标签并编号   将Dataset的数据进行打包

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

img_path = 'data/hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(img_path)
img.show()

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_itm_path = os.path.join(self.path, img_name)
        img = Image.open(img_itm_path)
        return img, self.label_dir

    def __len__(self):
        return len(self.img_path)

ants_dataset = MyData('data/hymenoptera_data/train', 'ants')
img, label = ants_dataset[0]
img.show()

bees_dataset = MyData('data/hymenoptera_data/train', 'bees')

train_dataset = ants_dataset + bees_dataset
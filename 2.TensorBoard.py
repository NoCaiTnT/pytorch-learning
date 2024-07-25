from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# 1. 实例化SummaryWriter
writer = SummaryWriter('logs')

# 2. 使用add_scalar()方法添加标量数据
# 参数为:
#   tag: 标题
#   scalar_value: y轴的值
#   global_step: x轴的值
for i in range(100):
    writer.add_scalar('y=2x', i * 2, i)

# 3. 打开终端, 运行以下命令:
#   tensorboard --logdir=logs --port=6007
#   然后在浏览器中输入: http://localhost:6007
#   就可以看到图
# 注意: 每次运行需要将logs文件夹中的东西删除, 否则会图像显示错误(同名的图会保留上一次的数据)

# 使用writer.add_image()添加图片
img1_path = 'data/练手数据集/train/ants_image/0013035.jpg'
img1_PIL = Image.open(img1_path)
img1_array = np.array(img1_PIL)
print(img1_array.shape)
writer.add_image("test", img1_array, 1, dataformats='HWC')

img2_path = 'data/练手数据集/train/bees_image/16838648_415acd9e3f.jpg'
img2_PIL = Image.open(img2_path)
img2_array = np.array(img2_PIL)
print(img2_array.shape)
writer.add_image("test", img2_array, 2, dataformats='HWC')

writer.close()
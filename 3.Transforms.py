# 图像预处理
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 1. ToTensor
img1_path = 'data/练手数据集/train/ants_image/0013035.jpg'
img1_PIL = Image.open(img1_path)
img1_tensor = transforms.ToTensor()(img1_PIL)
print(img1_tensor.shape)

writer = SummaryWriter('logs')
writer.add_image("Tensor_img", img1_tensor)

# 2. Normalize
img1_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img1_tensor)
writer.add_image("Norm_img", img1_norm)

# 3. Resize
print(img1_PIL.size)
img1_resize = transforms.Resize((512, 512))(img1_PIL)
img1_resize_tensor = transforms.ToTensor()(img1_resize)
writer.add_image("Resize_img", img1_resize_tensor)

# 4. Compose
#   将多个transform组合在一起
compose = transforms.Compose([
              transforms.Resize((300, 300)),
              transforms.ToTensor()])
img1_compose = compose(img1_PIL)
writer.add_image("Compose_img", img1_compose)

# 5. RandomCrop
compose2 = transforms.Compose([
              transforms.RandomCrop((300, 300)),
              transforms.ToTensor()])
for i in range(10):
    img1_randomcrop = compose2(img1_PIL)
    writer.add_image("RandomCrop_img", img1_randomcrop, i)

writer.close()
import torch
import torchvision
from PIL import Image
from MyCIFAR10 import MyCIFAR10

dog_path = "imgs/dog.png"
dog_img = Image.open(dog_path)
dog_img = dog_img.convert('RGB')

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.ToTensor()
])

dog_img = transform(dog_img)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = MyCIFAR10()
net.load_state_dict(torch.load('models/cifar10_10.pth'))
net.to(device)

net.eval()
with torch.no_grad():
    output = net(dog_img.unsqueeze(0).to(device))
print(output)
print(output.argmax(1))

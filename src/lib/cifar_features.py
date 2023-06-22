import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms

def getCIFAR10Data():
    PATH  =  os.path.dirname(os.path.abspath(__file__))
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 10

    data_path = os.path.join(PATH, '..', 'data')
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # frog  truck truck deer  car   car   bird  horse ship  cat

    return images, labels
    


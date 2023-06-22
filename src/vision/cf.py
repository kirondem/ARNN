import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 10

    data_path = os.path.join(PATH, '..', '..', 'data')
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    #imshow(torchvision.utils.make_grid(images))

    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # frog  truck truck deer

    model = models.resnet18(weights='IMAGENET1K_V1')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    img = images[0]

    img = torch.unsqueeze(img, 0)

    with torch.no_grad():
            x = 1
            outputs = model(img)

if __name__ == '__main__':
    main()
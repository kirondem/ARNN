import torchvision.models as models
import torchvision
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import utils
from PIL import Image
from torchvision import transforms


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow, rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == "__main__":

    device = torch.device('cpu')
    #if torch.cuda.is_available():
        #device = torch.device('cuda')

    # create vgg16 backbone
    model = torchvision.models.resnet50(pretrained=True)
    model.out_channels = 512
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )])

    img = Image.open("C:\\Work\\PhD\\src\\sandbox\\dog.jpg")
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    #output = model(batch_t)

    print(model)

    filter = model.layer4[2].conv3
    filter = filter.weight.data.clone()

    visTensor(filter, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()
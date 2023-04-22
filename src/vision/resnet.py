
import os
import cv2
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import numpy as np
from torchvision import datasets, models, transforms

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features

model = torch.nn.Sequential(*(list(model.children())[:-1]))

print(model)

model.eval()

# Load an image
img = cv2.imread(os.path.join(PATH, '..', 'data', 'shapes','circle.jpg'), 0) 
#img = Image.open(os.path.join(PATH, '..', 'data', 'dog.jpg'))
#img = img.resize((224, 224))
img = np.array(img)

# Convert to tensor
img = torch.from_numpy(img)
img = torch.unsqueeze(img, 0)

#img = img.permute(2, 0, 1) # to channel first

img = img[None, :, :, :] # to create the batch dimension
img = img / 255.0

#transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

with torch.no_grad():
        x = 1
        outputs = model(img)
        print(outputs.shape)
        print(outputs)
        #for i, (inputs, labels) in enumerate(dataloaders['val']):
            #inputs = inputs.to(device)
            #labels = labels.to(device)
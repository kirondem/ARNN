import logging
import argparse
import datetime
from operator import mod
import os
import random
import sys
import cv2
import torch
from torchvision import datasets, models, transforms
import torchvision

sys.path.append(os.getcwd())

import time
import numpy as np
from lib import enums, constants, utils, plot_utils, cifar_features
from models.associative_network import AssociativeNetwork
from lib.utils import concat_images, dynamic_lambda
from lib.activation_functions import relu

LOG_LEVEL = logging.getLevelName(constants.lOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger('matplotlib.font_manager').disabled = True

PATH  =  os.path.dirname(os.path.abspath(__file__))
APPLICATION = enums.Application.base_line.value

def get_hidden_features(x, model, layer):
    activation = {}

    def get_activation(name):
        def hook(m, i, o):
            activation[name] = o.detach()

        return hook

    model.register_forward_hook(get_activation(layer))
    _ = model(x)
    return activation[layer]


def main():

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1

    data_path = os.path.join(PATH, 'data')
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    logging.info('Loading model')
    model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    logging.info('Loading model completed')
    #model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(model)
    model.eval()
    #print(model[5][0].conv1.out_channels)

    children_counter = 0
    for n,c in model.named_children():
        print("Children Counter: ",children_counter," Layer Name: ",n,)
        children_counter+=1

    # output = get_hidden_features(images, model, "conv1")
    

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.conv1.register_forward_hook(get_activation('conv1'))
    output = model(images)
    print(output)
    output = activation['conv1']
    print(output)
    print(output.shape)
    f1 = output.view(output.size(0), -1)
    print(f1.shape)


if __name__ == '__main__':
    main()

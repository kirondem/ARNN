import logging
import argparse
import datetime
import os
import random
import sys
import cv2
import torch
from torchvision import models
from torchvision.models import resnet50, ResNet18_Weights
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
import time
import numpy as np
from lib import enums, constants, utils, plot_utils, cifar_features
from models.associative_network import AssociativeNetwork
from lib.utils import concat_images, dynamic_lambda, transform_inputs
from lib.activation_functions import relu, sigmoid
from models.auto_encoder import Autoencoder
import torch.nn as nn
import torch.optim as optim

LOG_LEVEL = logging.getLevelName(constants.lOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger('matplotlib.font_manager').disabled = True

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

PATH  =  os.path.dirname(os.path.abspath(__file__))
APPLICATION = enums.Application.base_line.value

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
activation = {}

def get_args_parser():
    parser = argparse.ArgumentParser('Train associative network', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--time_steps', default=constants.TIME_STEPS, type=int)
    parser.add_argument('--no_of_units', default=784, type=int, help='Number of units in the associative network')
    parser.add_argument('--no_of_input_units', default=784, type=int, help='Number of input units')
    
    return parser.parse_args()

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def train(network, data, data_size, batch_size, epochs, time_steps, lr, decay_threshold):

    for epoch in range(epochs):

        # decay learning rate
        learning_rate = lr * (1 - epoch / epochs)
        logging.info('Epoch {}, lr {}'.format( epoch, learning_rate))
        #data = data.reshape((1, data.shape[0]))
        
        # Iterate over data.
        for i in range(data_size//batch_size):
            
            input = data[i*batch_size: (i+1)*batch_size].flatten()
            
            W, Wt_LAST, H, H_H = network.learn(input, time_steps, learning_rate, decay_threshold)

            # Reseting the network timesteps
            network.reset(time_steps)
            network.init_weights(Wt_LAST.copy())

        H_H = H_H[-1]

    return W, Wt_LAST, H, H_H

def main():

    args = get_args_parser()
    trials = constants.TRIAL_EPOCHS
    decay_threshold = constants.DECAY_THRESHOLD
    N = 784
    data_size = 1 # 60000
    batch_size = 1
    network_type = enums.ANNNetworkType.DynamicLambda.value
    resized_image_dim = 16
    no_of_units_network = (resized_image_dim * resized_image_dim) * 2

    # Class names for the CIFAR-10 dataset
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    images, labels = cifar_features.getCIFAR10Data()
    images = images.to(device)
    
    # 10, 3, 32, 32
    # 6, 9, 9, 4, 1, 1, 2, 7, 8, 3
    # frog, truck, truck, deer, car, car, bird, horse, ship, cat

    #model = models.resnet18(weights='IMAGENET1K_V1')
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.layer3[1].register_forward_hook(get_activation('l3_conv2'))

    with torch.no_grad():
        output = model(images)
        features = activation['conv1'].cpu().numpy()
    
    features_frog = features[0][1]
    features_truck = features[1][1]
    features_bird = features[6][1]
    features_cat = features[9][1]

    s1 = np.concatenate((features_frog, np.zeros_like(features_cat)), axis=1)
    s2 = np.concatenate((features_cat, np.zeros_like(features_cat)), axis=1)
    s3 = np.concatenate((features_truck, np.zeros_like(features_cat)), axis=1)
    s4 = np.concatenate((features_bird, np.zeros_like(features_cat)), axis=1)

    s1 = s1.flatten().reshape((1, -1))
    s2 = s2.flatten().reshape((1, -1))
    s3 = s3.flatten().reshape((1, -1))
    s4 = s4.flatten().reshape((1, -1))

    ss = [s1, s2, s3, s4]

    
    auto_encoder = Autoencoder(512, 256).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(auto_encoder.parameters(), lr=0.001)
    
    ts = 5 # args.time_steps
    
    losses = []

    start_time = time.time()
    epochs = constants.AE_TRAIN_EPOCHS
    for i in range(epochs):
        print(f"Epoch {i+1} of {epochs}")
        Hs = []
        for s in ss:
            network = AssociativeNetwork(no_of_units_network, ts)
            _, Wt_LAST, H1, H_H1 = train(network, s, data_size, batch_size, args.epochs, ts, args.lr, decay_threshold)
            Hs.append(H_H1)
        
        Hs_array = np.array(Hs, dtype='float32')
        # Randomly shuffle the array
        np.random.shuffle(Hs_array)
        H_H1 = torch.from_numpy(Hs_array).to(device)

        optimizer.zero_grad()
        outputs = auto_encoder(H_H1)
        loss = criterion(outputs, H_H1)

        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
        losses.append(loss.item())

    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")

    # Save model weights
    path = os.path.join(PATH, 'saved_weights', f'autoencoder_{epochs}.pth')
    torch.save(auto_encoder.state_dict(), path)

    # Plot losses
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    main()
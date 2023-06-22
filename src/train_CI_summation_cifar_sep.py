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
from torchvision.models import resnet50, ResNet18_Weights

sys.path.append(os.getcwd())

import time
import numpy as np
from lib import enums, constants, utils, plot_utils, cifar_features
from models.associative_network import AssociativeNetwork
from lib.utils import concat_images, dynamic_lambda, transform_inputs
from lib.activation_functions import relu, sigmoid

LOG_LEVEL = logging.getLevelName(constants.lOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger('matplotlib.font_manager').disabled = True

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

def save_weights(Path, name, data, trials, epochs, time_steps, network_type):
    path = os.path.join(Path, 'saved_weights', '{}_{}_{}_{}_{}.npy'.format(trials, epochs, time_steps, name, network_type))
    with open(path, 'wb') as f:
        np.save(f, data)

def train_all(network1, network_assoc, s1 , s2, data_size, batch_size, args, decay_threshold):

    Wt_LAST_ASSOC = None
    logging.info("--Training network 1 -- S1")
    _, Wt_LAST, H1, H_H1 = train(network1, s1, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
    Wt_LAST = Wt_LAST.copy()

    network1.reset(args.time_steps)
    network1.init_weights(Wt_LAST)

    logging.info("--Training network 1 -- S2")
    _, Wt_LAST, H2, H_H2 = train(network1, s2, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
    Wt_LAST = Wt_LAST.copy()

    assoc_input = np.concatenate([H_H1, H_H2])
    assoc_input = assoc_input.reshape((1, assoc_input.shape[0]))

    # Pass through an Relu activation function
    ## assoc_input = relu(assoc_input)
    ## assoc_input = sigmoid(assoc_input)

    sum_activities = assoc_input.sum(axis=1)
    print(sum_activities)

    assoc_input = transform_inputs(assoc_input)

    logging.info("--Training assoc network H_H1 + H_H2")
    _, Wt_LAST_ASSOC, H3, H_H3 = train(network_assoc, assoc_input, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
    Wt_LAST_ASSOC = Wt_LAST_ASSOC.copy()

    # Reset networks and restore weights

    network1.reset(args.time_steps)
    network1.init_weights(Wt_LAST)

    network_assoc.reset(args.time_steps)
    network_assoc.init_weights(Wt_LAST_ASSOC)

    logging.info("Finished training associative network")

    return Wt_LAST, Wt_LAST_ASSOC 

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    args = get_args_parser()

    trials = constants.TRIAL_EPOCHS
    decay_threshold = constants.DECAY_THRESHOLD
    N = 784
    data_size = 1 # 60000
    batch_size = 1
    
    network_type = enums.ANNNetworkType.DynamicLambda.value

    #Read in fashion mnist data
    resized_image_dim = 16

    images, labels = cifar_features.getCIFAR10Data()
    images = images.to(device)
    # frog  truck truck deer  car   car   bird  horse ship  cat

    #model = models.resnet18(weights='IMAGENET1K_V1')
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()

    model.conv1.register_forward_hook(get_activation('conv1'))

    with torch.no_grad():
        output = model(images)
        features = activation['conv1'].cpu().numpy()

    features_frog = features[0][1]
    features_truck = features[1][1]
    features_bird = features[6][1]
    features_cat = features[9][1]

    # Make are all features that negative are zero
    features_frog[features_frog < 0] = 0
    features_truck[features_truck < 0] = 0
    features_bird[features_bird < 0] = 0
    features_cat[features_cat < 0] = 0
    
    #Nothing image
    nothing_image = np.zeros((resized_image_dim, resized_image_dim))
    nothing_image_double = concat_images(nothing_image, nothing_image)

    no_of_units_network_1 = (resized_image_dim * resized_image_dim) * 2
    no_of_units_network_2 = resized_image_dim * resized_image_dim

    network1 = AssociativeNetwork(no_of_units_network_1, args.time_steps)
    network_assoc = AssociativeNetwork(no_of_units_network_1 * 2 , args.time_steps)

    start_time = time.time()
    logging.info("Start training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))

    for i in range(0):
        logging.info("🚀 Trial: {} 🚀".format(i+1))
        # Cross -> Star
        # Frog -> Cat
        s1 = concat_images(features_frog, nothing_image)
        s2 = concat_images(features_cat, nothing_image)

        s1 = s1.flatten()
        s1 = s1.reshape((1, s1.shape[0]))
        s2 = s2.flatten()
        s2 = s2.reshape((1, s2.shape[0]))
        Wt_LAST, Wt_LAST_ASSOC = train_all(network1, network_assoc, s1, s2, data_size, batch_size, args, decay_threshold)

    #save_weights(PATH, 'Wt_LAST1_1', Wt_LAST, trials, args.epochs, args.time_steps, network_type)
    #save_weights(PATH, 'Wt_LAST_ASSOC_1', Wt_LAST_ASSOC, trials, args.epochs, args.time_steps, network_type)

    for i in range(trials):
        logging.info("🚀 Trial: {} 🚀".format(i+1))
        # Circle -> Star
        # Truck -> Cat
        s1 = concat_images(features_truck, nothing_image)
        s2 = concat_images(features_cat, nothing_image)

        s1 = s1.flatten()
        s1 = s1.reshape((1, s1.shape[0]))
        s2 = s2.flatten()
        s2 = s2.reshape((1, s2.shape[0]))

        Wt_LAST, Wt_LAST_ASSOC = train_all(network1, network_assoc, s1, s2, data_size, batch_size, args, decay_threshold)

        # Circle + Arrow -> NOTHING
        # Truck + Bird  -> NOTHING
        s1 = concat_images(features_truck, features_bird)
        s2 = nothing_image_double

        s1 = s1.flatten()
        s1 = s1.reshape((1, s1.shape[0]))
        s2 = s2.flatten()
        s2 = s2.reshape((1, s2.shape[0]))

        Wt_LAST, Wt_LAST_ASSOC = train_all(network1, network_assoc, s1, s2, data_size, batch_size, args, decay_threshold)

    end_time = time.time()

    save_weights(PATH, 'Wt_LAST1_2', Wt_LAST, trials, args.epochs, args.time_steps, network_type)
    save_weights(PATH, 'Wt_LAST_ASSOC_2', Wt_LAST_ASSOC, trials, args.epochs, args.time_steps, network_type)
    
    logging.info("End training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logging.info("Training took {0:.1f}".format(end_time-start_time))

if __name__ == '__main__':
    main()
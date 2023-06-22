import logging
import argparse
import datetime
from operator import mod
import os
import random
import sys
import cv2
sys.path.append(os.getcwd())
import time
import numpy as np
from lib import enums, constants, utils, plot_utils, cifar_features
from models.associative_network import AssociativeNetwork
from lib.utils import concat_images, dynamic_lambda, load_weights, save_weights, transform_inputs
from lib.activation_functions import relu
import torch
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet18_Weights

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

    start_time = time.time()

    logging.info("Start training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))

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
            network.init_weights(Wt_LAST)

        H_H = H_H[-1]


    end_time = time.time()

    logging.info("End training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logging.info("Training took {0:.1f}".format(end_time-start_time))

    return W, Wt_LAST, H, H_H

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

    network1 = AssociativeNetwork(no_of_units_network_1, args.time_steps)
    network_assoc = AssociativeNetwork(no_of_units_network_1 * 2 , args.time_steps)

    # Load the saved weights
    Wt_LAST1 = load_weights(PATH, 'Wt_LAST1_2', trials, args.epochs, args.time_steps, network_type)
    Wt_LAST_ASSOC = load_weights(PATH, 'Wt_LAST_ASSOC_2', trials, args.epochs, args.time_steps, network_type)

    network1.init_weights(Wt_LAST1)
    network_assoc.init_weights(Wt_LAST_ASSOC)

    #TEST 1
    # CS+ -> US
    # Cross + Arrow -> Star
    # Frog + Bird -> Cat

    s1 = concat_images(features_frog, features_bird)
    s2 = concat_images(features_cat, nothing_image)

    s1 = s1.flatten()
    s1 = s1.reshape((1, s1.shape[0]))
    s2 = s2.flatten()
    s2 = s2.reshape((1, s2.shape[0]))


    logging.info("--Training network 1 -- S1")
    _, Wt_LAST, H1, H_H1 = train(network1, s1, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)


    network1.reset(args.time_steps)
    network1.init_weights(Wt_LAST.copy())

    logging.info("--Training network 1 -- S2")
    _, Wt_LAST, H2, H_H2 = train(network1, s2, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)

    assoc_input = np.concatenate([H_H1, H_H2])
    assoc_input = assoc_input.reshape((1, assoc_input.shape[0]))
    print(assoc_input.shape)

    assoc_input = transform_inputs(assoc_input)

    logging.info("--Training assoc network H_H1 + H_H2")
    _, Wt_LAST_ASSOC, H3, H_H3 = train(network_assoc, assoc_input, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
    Wt_LAST_ASSOC = Wt_LAST_ASSOC.copy()

    print(Wt_LAST_ASSOC.shape)

    total_last_assoc_weights = []
    for from_idx in range(H_H1.shape[0]):
        for to_idx in range(H_H1.shape[0]):
            total_last_assoc_weights.append( Wt_LAST_ASSOC[from_idx, to_idx])
            
    total_last_assoc_weights = np.array(total_last_assoc_weights)
    total_last_assoc_weights = np.dot(total_last_assoc_weights.T, total_last_assoc_weights)
    print('total_last_assoc_weights: ', total_last_assoc_weights)

    total_activations = []
    for from_idx in range(H_H1.shape[0]):
        for to_idx in range(H_H1.shape[0]):
            total_activations.append( Wt_LAST_ASSOC[from_idx, to_idx] * H_H1[from_idx])
            
    total_activations = np.array(total_activations)
    sum_activities = np.dot(total_activations.T, total_activations)
    print('sum_activities: ', sum_activities)

    x = 1

if __name__ == '__main__':
    main()
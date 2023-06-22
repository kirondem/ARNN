import logging
import argparse
import datetime
from operator import mod
import os
import random
import sys
sys.path.append(os.getcwd())
import time
import numpy as np
import cv2
from lib import enums, constants, utils, plot_utils
from models.associative_network import AssociativeNetwork
from lib.utils import concat_images, dynamic_lambda, lambda_US_magnitude
from lib.activation_functions import relu

LOG_LEVEL = logging.getLevelName(constants.lOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger('matplotlib.font_manager').disabled = True

PATH  =  os.path.dirname(os.path.abspath(__file__))
APPLICATION = enums.Application.base_line.value

def get_args_parser():
    parser = argparse.ArgumentParser('Train associative network', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
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


def save_weights(Path, name, data, trials, epochs, time_steps):
    path = os.path.join(Path, 'saved_weights', '{}_{}_{}_{}.npy'.format(trials, epochs, time_steps, name))
    with open(path, 'wb') as f:
        np.save(f, data)

def load_weights(Path, name, trials, epochs, time_steps, network_type):
    path = os.path.join(Path, 'saved_weights', '{}_{}_{}_{}_{}.npy'.format(trials, epochs, time_steps, name, network_type))
    with open(path, 'rb') as f:
        data = np.load(f)
    return data

def main():
    args = get_args_parser()

    #Read in fashion mnist data
    # resized_image_dim = constants.
    resized_image_dim = 16
    no_of_units_network_1 = (resized_image_dim * resized_image_dim) * 2
    network_assoc = AssociativeNetwork(no_of_units_network_1 * 2 , args.time_steps)

    # Load the saved weights
    path = os.path.join(PATH, 'saved_weights', '20_1_4_Wt_LAST1_dynamic_lambda_1.npy')
    # path = os.path.join(PATH, 'saved_weights', 'C-B_50_1_4_Wt_LAST_ASSOC_dynamic_lambda.npy')
    
    with open(path, 'rb') as f:
        Wt_LAST_ASSOC = np.load(f)
    
    network_assoc.init_weights(Wt_LAST_ASSOC)
    print(Wt_LAST_ASSOC.shape)

    total_weights = []
    for from_idx in range(0, int(Wt_LAST_ASSOC.shape[0] / 2)):
        for to_idx in range(int(Wt_LAST_ASSOC.shape[0] / 2), Wt_LAST_ASSOC.shape[0]):
            total_weights.append( Wt_LAST_ASSOC[from_idx, to_idx] )
            
    total_weights = np.array(total_weights)
    sum_activities = total_weights.sum(axis=0)
    print(sum_activities)


    x = 1

if __name__ == '__main__':
    main()
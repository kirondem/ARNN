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

    #TODO: REMOVE
    #args.time_steps = 2

    trials = constants.TRIAL_EPOCHS
    decay_threshold = constants.DECAY_THRESHOLD
    N = 784
    data_size = 1 # 60000
    batch_size = 1

    network_type = enums.ANNNetworkType.DynamicLambda.value

    #Read in fashion mnist data
    resized_image_dim = constants.RESIZED_IMAGE_DIM

    img_circle = cv2.imread(os.path.join(PATH,  'data', 'shapes','circle.jpg'), 0) 
    img_circle = cv2.resize(img_circle, (resized_image_dim, resized_image_dim))
    img_circle = img_circle / constants.INPUT_SCALING_FACTOR
    
    img_star = cv2.imread(os.path.join(PATH,  'data', 'shapes','star.jpg'), 0) 
    img_star = cv2.resize(img_star, (resized_image_dim, resized_image_dim))
    img_star = img_star / constants.INPUT_SCALING_FACTOR

    img_arrow = cv2.imread(os.path.join(PATH,  'data', 'shapes','arrow.jpg'), 0) 
    img_arrow = cv2.resize(img_arrow, (resized_image_dim, resized_image_dim))
    img_arrow = img_arrow / constants.INPUT_SCALING_FACTOR

    img_grid = cv2.imread(os.path.join(PATH,  'data', 'shapes','arrow.jpg'), 0) 
    img_grid = cv2.resize(img_grid, (resized_image_dim, resized_image_dim))
    img_grid = img_grid / constants.INPUT_SCALING_FACTOR

    img_cross = cv2.imread(os.path.join(PATH,  'data', 'shapes','cross.jpg'), 0) 
    img_cross = cv2.resize(img_cross, (resized_image_dim, resized_image_dim))
    img_cross = img_cross / constants.INPUT_SCALING_FACTOR

    #Nothing image
    nothing_image = np.zeros((resized_image_dim, resized_image_dim))
    nothing_image_double = concat_images(nothing_image, nothing_image)

    no_of_units_network_1 = (resized_image_dim * resized_image_dim) * 2
    no_of_units_network_2 = resized_image_dim * resized_image_dim

    network1 = AssociativeNetwork(no_of_units_network_1, args.time_steps)
    network_assoc = AssociativeNetwork(no_of_units_network_1 * 2 , args.time_steps)

    # Load the saved weights
    Wt_LAST1 = load_weights(PATH, 'Wt_LAST1', trials, args.epochs, args.time_steps, network_type)
    Wt_LAST_ASSOC = load_weights(PATH, 'Wt_LAST_ASSOC', trials, args.epochs, args.time_steps, network_type)

    network1.init_weights(Wt_LAST1)
    network_assoc.init_weights(Wt_LAST_ASSOC)

    #TEST 1
    # CS+ -> US
    # Cross -> Star


    s1 = concat_images(img_cross, nothing_image)
    s2 = concat_images(img_star, nothing_image)

    s1 = s1.flatten()
    s1 = s1.reshape((1, s1.shape[0]))
    s2 = s2.flatten()
    s2 = s2.reshape((1, s2.shape[0]))


    logging.info("--Training network 1 -- S1")
    _, Wt_LAST, H1, H_H1 = train(network1, s1, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
    Wt_LAST_1 = Wt_LAST.copy()

    network1.reset(args.time_steps)
    network1.init_weights(Wt_LAST.copy())

    logging.info("--Training network 1 -- S2")
    _, Wt_LAST, H2, H_H2 = train(network1, s2, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
    Wt_LAST_2 = Wt_LAST.copy()

    assoc_input = np.concatenate([H_H1, H_H2])
    assoc_input = assoc_input.reshape((1, assoc_input.shape[0]))
    print(assoc_input.shape)

    # Pass through an Relu activation function
    #assoc_input = relu(assoc_input)

    logging.info("--Training assoc network H_H1 + H_H2")
    _, Wt_LAST_ASSOC, H3, H_H3 = train(network_assoc, assoc_input, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
    Wt_LAST_ASSOC = Wt_LAST_ASSOC.copy()

    print(Wt_LAST_ASSOC.shape)

    total_activations = []
    for from_idx in range(H_H1.shape[0]):
        for to_idx in range(H_H1.shape[0]):
            total_activations.append( Wt_LAST_ASSOC[from_idx, to_idx] * H_H1[from_idx])
            
    total_activations = np.array(total_activations)
    sum_activities = np.dot(total_activations.T, total_activations)
    print(sum_activities)


    x = 1

if __name__ == '__main__':
    main()
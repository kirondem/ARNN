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
from lib import enums, constants, utils, plot_utils
from models.associative_network import AssociativeNetwork
from lib.utils import concat_images, dynamic_lambda
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
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--time_steps', default=4, type=int)
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
            network.init_weights(Wt_LAST)

        H_H = H_H[-1]

    return W, Wt_LAST, H, H_H

def associate_inputs(W_ASSOC, learning_rate, input, target):
    Zh = np.dot(input.reshape(1, -1), np.transpose(W_ASSOC))
    Zh = Zh.flatten()

    y = target.flatten()
    #print(y)
    for from_idx in range(input.shape[0]):
        for to_idx in range(y.shape[0]):
            h_to = y[to_idx]
            h_from = input[from_idx]
            
            # 3) Calculate maximum conditioning possible for the US
            to_lambda_max = dynamic_lambda(h_from, h_to)

            v_total = Zh[to_idx]
            d_w = learning_rate * h_from * (to_lambda_max * h_to - (v_total))

            if 1 == 2:
                print('------------------')
                print("lambda max:", to_lambda_max)
                print("v_total:", v_total)
                print("d_w:", d_w)

            w = W_ASSOC[to_idx, from_idx]
            W_ASSOC[to_idx][from_idx] = W_ASSOC[to_idx, from_idx] + d_w
 
    return W_ASSOC

def associate(W_A, network1, network2, data1, data2, data_size, batch_size, epochs, time_steps, lr, decay_threshold):

    start_time = time.time()
    logging.info("Start associate {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))

    for epoch in range(epochs):

        # decay learning rate
        learning_rate = lr * (1 - epoch / epochs)
        logging.info('Epoch {}, lr {}'.format( epoch, learning_rate))
        #data = data.reshape((1, data.shape[0]))
        
        # Iterate over data.
        for i in range(data_size//batch_size):
            
            input = data1[i*batch_size: (i+1)*batch_size].flatten()
            W, Wt_LAST1, H, H_H1 = network1.learn(input, time_steps, learning_rate, decay_threshold)

            input = data2[i*batch_size: (i+1)*batch_size].flatten()
            W, Wt_LAST2, H, H_H2 = network2.learn(input, time_steps, learning_rate, decay_threshold)

            W_A = associate_inputs(W_A, learning_rate, H_H1[-1], H_H2[-1])

            network1.reset(time_steps)
            network1.init_weights(Wt_LAST1)

            network2.reset(time_steps)
            network2.init_weights(Wt_LAST2)
            
    end_time = time.time()

    logging.info("End associate {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logging.info("associate took {0:.1f}".format(end_time-start_time))

    return W_A

def save_weights(Path, name, data, trials, epochs, time_steps, network_type):
    path = os.path.join(Path, 'saved_weights', '{}_{}_{}_{}_{}.npy'.format(trials, epochs, time_steps, name, network_type))
    with open(path, 'wb') as f:
        np.save(f, data)

def train_all(network1, network_assoc, s1 , s2, data_size, batch_size, args, decay_threshold):

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
    assoc_input = relu(assoc_input)

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

def main():

    args = get_args_parser()
    trials = 1
    N = 784
    data_size = 1 # 60000
    batch_size = 1
    decay_threshold = 0.1
    network_type = enums.ANNNetworkType.DynamicLambda.value

    #Read in fashion mnist data
    resized_image_dim = 15
    X_fashion_mnist_data = np.zeros((0, N))

    path = os.path.join(PATH, 'data', 'fashion-mnist','images-idx3-ubyte.npy')
    with open(path, 'rb') as f:
        X_fashion_mnist_data = np.load(f)
    
    # y_fashion_mnist_data = np.zeros((0, N))
    # path = os.path.join(PATH, 'data', 'fashion-mnist','labels-idx1-ubyte.npy')
    # with open(path, 'rb') as f:
    #     y_fashion_mnist_data = np.load(f)

    # Fashion labels 9, 3, 4, 5, 8, 4

    #handbags_indexes = [35, 57, 99, 100]
    handbags_indexes = [35]

    #sandles_indexes = [9, 12, 13, 30 ]
    sandles_indexes = [9]

    #dresses_indexes = [1064, 1077, 1093, 1108, 1115, 1120]
    dresses_indexes = [1064]

    #Trousers indexes
    #trousers_indexes = [21, 38, 69, 71]
    trousers_indexes = [21]
    
    
    X_fashion_mnist_data_handbags = X_fashion_mnist_data[handbags_indexes] / 255.0
    X_fashion_mnist_data_dresses = X_fashion_mnist_data[dresses_indexes] / 255.0
    X_fashion_mnist_data_sandles = X_fashion_mnist_data[sandles_indexes] / 255.0
    X_fashion_mnist_data_trousers = X_fashion_mnist_data[trousers_indexes] / 255.0

    #Nothing image
    nothing_image = np.zeros((resized_image_dim, resized_image_dim))
    nothing_image_double = concat_images(nothing_image, nothing_image)

    no_of_units_network_1 = (resized_image_dim * resized_image_dim) * 2
    no_of_units_network_2 = resized_image_dim * resized_image_dim

    network1 = AssociativeNetwork(no_of_units_network_1, args.time_steps)
    network_assoc = AssociativeNetwork(no_of_units_network_1 * 2 , args.time_steps)

    start_time = time.time()

    logging.info("Start training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))

    for i in range(trials):
        logging.info("🚀 Trial: {} 🚀".format(i+1))
        
        # Handbags -> sandles
        img1 = random.sample(list(X_fashion_mnist_data_handbags), 1)[0].reshape(28,28)
        img1 = cv2.resize(img1, (resized_image_dim, resized_image_dim))
        img2 = nothing_image

        s1 = concat_images(img1, img2)
        s2 = random.sample(list(X_fashion_mnist_data_sandles), 1)[0].reshape(28,28)
        s2 = cv2.resize(s2, (resized_image_dim, resized_image_dim))
        s2 = concat_images(s2, nothing_image)

        s1 = s1.flatten()
        s1 = s1.reshape((1, s1.shape[0]))
        s2 = s2.flatten()
        s2 = s2.reshape((1, s2.shape[0]))

        Wt_LAST, Wt_LAST_ASSOC = train_all(network1, network_assoc, s2, s2, data_size, batch_size, args, decay_threshold)
        
        # Handbags + dresses -> NOTHING
        img1 = random.sample(list(X_fashion_mnist_data_handbags), 1)[0].reshape(28,28)
        img1 = cv2.resize(img1, (resized_image_dim, resized_image_dim))
        img2 = random.sample(list(X_fashion_mnist_data_dresses),1 )[0].reshape(28,28)
        img2 = cv2.resize(img2, (resized_image_dim, resized_image_dim))

        s1 = concat_images(img1, img2)
        s2 = nothing_image_double

        s1 = s1.flatten()
        s1 = s1.reshape((1, s1.shape[0]))
        s2 = s2.flatten()
        s2 = s2.reshape((1, s2.shape[0]))
        Wt_LAST, Wt_LAST_ASSOC = train_all(network1, network_assoc, s2, s2, data_size, batch_size, args, decay_threshold)

        # Trousers -> sandles
        img1 = random.sample(list(X_fashion_mnist_data_trousers), 1)[0].reshape(28,28)
        img1 = cv2.resize(img1, (resized_image_dim, resized_image_dim))
        img2 = nothing_image

        s1 = concat_images(img1, img2)
        s2 = random.sample(list(X_fashion_mnist_data_sandles), 1)[0].reshape(28,28)
        s2 = cv2.resize(s2, (resized_image_dim, resized_image_dim))
        s2 = concat_images(s2, nothing_image)

        s1 = s1.flatten()
        s1 = s1.reshape((1, s1.shape[0]))
        s2 = s2.flatten()
        s2 = s2.reshape((1, s2.shape[0]))
        Wt_LAST, Wt_LAST_ASSOC = train_all(network1, network_assoc, s2, s2, data_size, batch_size, args, decay_threshold)

        save_weights(PATH, 'Wt_LAST1', Wt_LAST, trials, args.epochs, args.time_steps, network_type)
        save_weights(PATH, 'Wt_LAST_ASSOC', Wt_LAST_ASSOC, trials, args.epochs, args.time_steps, network_type)

    end_time = time.time()

    logging.info("End training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logging.info("Training took {0:.1f}".format(end_time-start_time))

    x = 1

if __name__ == '__main__':
    main()
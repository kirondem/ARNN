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
from lib import enums, constants, utils, plot_utils
from models.associative_network import AssociativeNetwork
from lib.utils import concat_images, dynamic_lambda

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

def associate_old(network, data, y, data_size, batch_size, time_steps, decay_threshold, learning_rate, Wt_LAST):
    for i in range(data_size//batch_size):
        input = data[i*batch_size: (i+1)*batch_size]
        input = input.flatten()

        label = y[i*batch_size: (i+1)*batch_size]
        print("---- Input " + str(i))
        
        Wt_LAST, network.associate(input, label, time_steps, learning_rate, decay_threshold)

        # Reseting the network timesteps
        #W_last = W[-1]
        network.reset(time_steps, Wt_LAST)

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
    #inference(network, data, y, data_size, batch_size, time_steps, decay_threshold, learning_rate, Wt_LAST)

def inference(network, data, y, data_size, batch_size, time_steps, decay_threshold, learning_rate, Wt_LAST):
    for i in range(data_size//batch_size):
        input = data[i*batch_size: (i+1)*batch_size]
        input = input.flatten()

        label = y[i*batch_size: (i+1)*batch_size]
        print("---- Input " + str(i))
        
        Wt_LAST, network.predict(input, label, time_steps, learning_rate, decay_threshold)

        # Reseting the network timesteps
        #W_last = W[-1]
        network.reset(time_steps, Wt_LAST)

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

def save_weights(Path, name, data, trials, epochs, time_steps):
    path = os.path.join(Path, 'saved_weights', '{}_{}_{}_{}.npy'.format(trials, epochs, time_steps, name))
    with open(path, 'wb') as f:
        np.save(f, data)

def main():

    args = get_args_parser()
    trials = 8
    N = 784
    data_size = 1 # 60000
    batch_size = 1
    decay_threshold = 0.1

    #Read in fashion mnist data
    X_fashion_mnist_data = np.zeros((0, N))

    path = os.path.join(PATH, 'data', 'fashion-mnist','images-idx3-ubyte.npy')
    with open(path, 'rb') as f:
        X_fashion_mnist_data = np.load(f)
    
    # y_fashion_mnist_data = np.zeros((0, N))
    # path = os.path.join(PATH, 'data', 'fashion-mnist','labels-idx1-ubyte.npy')
    # with open(path, 'rb') as f:
    #     y_fashion_mnist_data = np.load(f)

    # #Fashion labels 9, 3, 4, 5, 8, 4

    handbags_indexes = [35, 57, 99, 100]
    #handbags_indexes = [35]

    sandles_indexes = [9, 12, 13, 30 ]
    #sandles_indexes = [9]

    dresses_indexes = [1064, 1077, 1093, 1108, 1115, 1120]
    #dresses_indexes = [1064]
    
    X_fashion_mnist_data_handbags = X_fashion_mnist_data[handbags_indexes] / 255.0
    X_fashion_mnist_data_dresses = X_fashion_mnist_data[dresses_indexes] / 255.0
    X_fashion_mnist_data_sandles = X_fashion_mnist_data[sandles_indexes] / 255.0

    #Nothing image
    nothing_image = np.zeros((28,28))

    no_of_units_network_1 = (28 * 28) * 2
    no_of_units_network_2 = 28 * 28

    W_A = np.zeros((no_of_units_network_2, no_of_units_network_1))

    network1 = AssociativeNetwork(no_of_units_network_1, args.time_steps)
    network2 = AssociativeNetwork(no_of_units_network_2, args.time_steps)

    for i in range(trials):
        logging.info("🚀 Trial: {} 🚀".format(i+1))
        if i % 2 == 0:
            # Handbags -> sandles
            img1 = random.sample(list(X_fashion_mnist_data_handbags), 1)[0]
            img2 = nothing_image
            s1 = concat_images(img1.reshape(28,28), img2)
            s2 = random.sample(list(X_fashion_mnist_data_sandles), 1)[0]
        else:
            # Handbags + dresses -> NOTHING
            img1 = random.sample(list(X_fashion_mnist_data_handbags), 1)[0]
            img2 = random.sample(list(X_fashion_mnist_data_dresses),1 )[0]
            s1 = concat_images(img1.reshape(28,28), img2.reshape(28,28))
            s2 = nothing_image

        #plot_utils.display_mult_images([s1, s1], ['',  ''], 1, 2)
        
        s1 = s1.flatten()
        s1 = s1.reshape((1, s1.shape[0]))
        s2 = s2.flatten()
        s2 = s2.reshape((1, s2.shape[0]))

        logging.info("--Training network 1 --")
        _, Wt_LAST1, H1, H_H1 = train(network1, s1, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
        Wt_LAST1 = Wt_LAST1.copy()

        #plot_utils.plot_H(H1[-1]) plot_utils.plot_H(H_H1)

        logging.info("--Training network 2--")
        _, Wt_LAST2, H2, H_H2 = train(network2, s2, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
        Wt_LAST2 = Wt_LAST2.copy()

        #plot_utils.plot_H(H2[-1])
        #plot_utils.plot_H(H_H2)

        network1.reset(args.time_steps)
        network1.init_weights(Wt_LAST1)

        network2.reset(args.time_steps)
        network2.init_weights(Wt_LAST2)

        logging.info("Training associative network")

        #for i in range(trials):
        #learning_rate = args.lr * (1 - i / trials)
        
        W_A = associate_inputs(W_A, args.lr, H_H1, H_H2)
    
    save_weights(PATH, 'Wt_LAST1', Wt_LAST1, trials, args.epochs, args.time_steps)
    save_weights(PATH, 'Wt_LAST2', Wt_LAST2, trials, args.epochs, args.time_steps)
    save_weights(PATH, 'W_A', W_A, trials, args.epochs, args.time_steps)

    x = 1

if __name__ == '__main__':
    main()
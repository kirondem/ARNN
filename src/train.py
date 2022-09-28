import logging
import argparse
import datetime
import os
import sys
sys.path.append(os.getcwd())
import scipy.io
import time
import numpy as np
from lib import enums, constants, plot_utils
from matplotlib import pyplot as plt
from models.associative_network import AssociativeNetwork

LOG_LEVEL = logging.getLevelName(constants.lOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger('matplotlib.font_manager').disabled = True

PATH  =  os.path.dirname(os.path.abspath(__file__))
APPLICATION = enums.Application.base_line.value

def get_args_parser():
    parser = argparse.ArgumentParser('Train associative network', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--time_steps', default=1, type=int)
    parser.add_argument('--no_of_units', default=484, type=int, help='Number of units in the associative network')
    parser.add_argument('--no_of_input_units', default=196, type=int, help='Number of input units')
    
    return parser.parse_args()

def gen_image(arr):

    small_image = arr.reshape((1, output_size, bin_size,output_size, bin_size)).max(4).max(2)
    small_image = np.squeeze(small_image, axis=0)

    #res = cv2.resize(arr, dsize=(14, 14), interpolation=cv2.INTER_CUBIC)
    
    plt.imshow(small_image, interpolation='nearest')
    return plt

def update(W, state,idx=None):
    w = W[idx]
    new_state = np.matmul(W[idx], state)
    state[idx] = new_state
    return state

def predict(W, state):
    iterations = 10
    for i in range(iterations):
        idx = np.random.randint(state.size)
        state = update(W, state, idx)
    return state

def show_image(small_image):
    small_image = np.expand_dims(small_image, axis=0)

    small_image = small_image.reshape((1, 14, 1, 14, 1)).max(4).max(2)

    small_image = np.squeeze(small_image, axis=0)

    plt.imshow(small_image, interpolation='nearest')

    plt.show()

def train(network, data, data_size, batch_size, epochs, time_steps, lr, decay_threshold, data_output_size, bin_size):

    start_time = time.time()

    logging.info("Start training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))

    for epoch in range(epochs):

        # decay learning rate
        learning_rate = lr * (1 - epoch / epochs)

        logging.info('Epoch {}, lr {}'.format( epoch, learning_rate))

        # Randomise the data
        #data = data[np.random.permutation(data_size),:] # Randomise the data

        data = data[0]
        data = data.reshape((1, data.shape[0]))

        # Iterate over data.
        for i in range(data_size//batch_size):
            
            input = data[i*batch_size: (i+1)*batch_size]
            small_image = input.reshape((1, data_output_size, bin_size, data_output_size, bin_size)).max(4).max(2)
            small_image = np.squeeze(small_image, axis=0)
            small_image = small_image.T
            small_image = small_image.flatten()

            ##show_image(small_image)

            #data = np.squeeze(data, axis=1)
            # W, H_H = network.learn(small_image, learning_rate)

            W, H, H_H, avg_w_list = network.learn(small_image, time_steps, learning_rate, decay_threshold)

        x_ticks = [str(i + 1) for i in range(len(avg_w_list))]
        plot_utils.save_plot(avg_w_list, "Avg weight", "Time steps", "Average weight", os.path.join(PATH, 'plots', "avg_w_{}.png".format(time_steps)), x_ticks)


        #show_image(small_image)
        path = os.path.join(PATH, 'saved_models', '{}_{}_h.npy'.format(epochs, time_steps))
        with open(path, 'wb') as f:
            np.save(f, H)

        path = os.path.join(PATH, 'saved_models', '{}_{}_hh.npy'.format(epochs, time_steps))
        with open(path, 'wb') as f:
            np.save(f, H_H)

        H_H = H_H[-1]
        #state = predict(W, small_image)
        state = H_H.reshape((1, 22, 1, 22, 1)).max(4).max(2)
        #state = state.reshape((1, 14, 1, 14, 1)).max(4).max(2)

        state = np.squeeze(state, axis=0)
        path = os.path.join(PATH, 'plots', '{}_{}_h.png'.format(epochs, time_steps))
        plt.clf()
        plot_utils.save_image(state, '', '', "Associative activation (H)", path)
        #plt.show()

        #gen_image(state).show()

    end_time = time.time()

    logging.info("End training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logging.info("Training took {0:.1f}".format(end_time-start_time))

def main():

    args = get_args_parser()

    Nc = 10
    N = 784
    data_size = 1 # 60000
    batch_size = 1
    decay_threshold = 0.1

    mat = scipy.io.loadmat(os.path.join(PATH, 'data', 'mnist_all.mat'))
    data = np.zeros((0, N))
    for i in range(Nc):
        data=np.concatenate((data, mat['train'+str(i)]), axis=0)
    
    data = data/255.0
    data_input_size = 28
    data_output_size = 14
    bin_size = data_input_size // data_output_size

    network = AssociativeNetwork(args.no_of_units, args.time_steps)

    train(network, data, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold, data_output_size, bin_size)

if __name__ == '__main__':
    main()
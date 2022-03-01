import logging
import argparse
import os
import scipy.io
import numpy as np
from PIL import Image
from lib import enums, constants, plot_utils
from configparser import ConfigParser
from matplotlib import pyplot as plt
from models.associative_network import AssociativeNetwork

LOG_LEVEL = logging.getLevelName(constants.lOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger('matplotlib.font_manager').disabled = True

PATH  =  os.path.dirname(os.path.abspath(__file__))
APPLICATION = enums.Application.base_line.value

config_parser = ConfigParser()
config_parser.read('configuration.ini')

Nc = 10
N = 784
data_size = 1 # 60000
batch_size = 1

input_size = 28
output_size = 14
bin_size = input_size // output_size
decay_threshold = 0.1

def get_args_parser():
    parser = argparse.ArgumentParser('Train associative network', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--time_steps', default=3, type=int)
    parser.add_argument('--no_of_units', default=131100, type=int, help='Number of units in the associative network')

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

def main():

    args = get_args_parser()

    im_frame = Image.open(os.path.join(PATH, 'data', 'faces', '330279.png'))
    data = np.array(im_frame)
    data = data.flatten()

    #131100

    network = AssociativeNetwork(args.no_of_units, args.time_steps)

    for epoch in range(args.epochs):

        # decay learning rate
        learning_rate = args.lr * (1 - epoch / args.epochs)

        # Randomise the data
        #data = data[np.random.permutation(data_size),:] # Randomise the data

        data = data[0]
        data = data.reshape((1, data.shape[0]))

        # Iterate over data.
        for i in range(data_size//batch_size):
            
            input = data[i*batch_size: (i+1)*batch_size]
            small_image = input.reshape((1, output_size, bin_size,output_size, bin_size)).max(4).max(2)
            small_image = np.squeeze(small_image, axis=0)
            small_image = small_image.T
            small_image = small_image.flatten()

            show_image(small_image)

            #data = np.squeeze(data, axis=1)
            # W, H_H = network.learn(small_image, learning_rate)

            W, H, H_H = network.learn(small_image, args.time_steps, learning_rate, decay_threshold)

        #show_image(small_image)

        H_H = H_H[-1]
        #state = predict(W, small_image)
        state = H_H.reshape((1, 22, 1, 22, 1)).max(4).max(2)
        #state = state.reshape((1, 14, 1, 14, 1)).max(4).max(2)

        state = np.squeeze(state, axis=0)
        path = os.path.join('plots', '{}_{}_h.png'.format(args.epochs, args.time_steps))
        plot_utils.save_image(state, '', '', "Associative activation (H)", path)
        plt.show()

        t=1

        #gen_image(state).show()
    

if __name__ == '__main__':
    main()
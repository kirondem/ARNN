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
from sklearn import preprocessing
from models.associative_network import AssociativeNetwork
import seaborn as sns

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
    parser.add_argument('--time_steps', default=2, type=int)
    parser.add_argument('--no_of_units', default=784, type=int, help='Number of units in the associative network')
    parser.add_argument('--no_of_input_units', default=784, type=int, help='Number of input units')
    
    return parser.parse_args()

def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest', cmap='gray_r')
    plt.show()

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

def train(network, data, y, data_size, batch_size, epochs, time_steps, lr, decay_threshold, data_output_size, bin_size, APPLICATION):

    start_time = time.time()

    logging.info("Start training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))

    for epoch in range(epochs):

        # decay learning rate
        learning_rate = lr * (1 - epoch / epochs)
        logging.info('Epoch {}, lr {}'.format( epoch, learning_rate))
        #data = data.reshape((1, data.shape[0]))
        
        # Iterate over data.
        for i in range(data_size//batch_size):
            
            input = data[i*batch_size: (i+1)*batch_size]
            input = input.flatten()

            label = y[i*batch_size: (i+1)*batch_size]
            

            W, W_OUT, H, H_H, avg_w_list, active_units_list, assoc_active_units_list = network.learn(input, label, time_steps, learning_rate, decay_threshold)

        x_ticks = [str(i + 1) for i in range(len(avg_w_list))]
        plot_utils.save_plot(avg_w_list, "Avg weight", "Time steps", "Average weight", os.path.join(PATH, 'plots', "avg_w_{}_{}.png".format(time_steps, APPLICATION)), x_ticks)
        
        x_ticks = [str(i + 1) for i in range(len(active_units_list))]
        plt.clf()
        plt.plot(list(range(0, len(active_units_list))), active_units_list)
        plt.plot(list(range(0, len(assoc_active_units_list))), assoc_active_units_list)
        plt.ylabel("No of active units")
        plt.xlabel("Time steps")
        if x_ticks is not None:
            plt.xticks(ticks=range(0,len(active_units_list)), labels=x_ticks)
        plt.legend(["Active units", "Assoc active units"])

        plt.title("No of unit activations")
        path = os.path.join(PATH, 'plots', "active_units_{}_{}.png".format(time_steps, APPLICATION))
        plt.savefig(path)

        path = os.path.join(PATH, 'saved_models', '{}_{}_{}_w.npy'.format(epochs, time_steps, APPLICATION))
        with open(path, 'wb') as f:
            np.save(f, W)

        path = os.path.join(PATH, 'saved_models', '{}_{}_{}_W_OUT.npy'.format(epochs, time_steps, APPLICATION))
        with open(path, 'wb') as f:
            np.save(f, W_OUT)
        
        path = os.path.join(PATH, 'saved_models', '{}_{}_{}_h.npy'.format(epochs, time_steps, APPLICATION))
        with open(path, 'wb') as f:
            np.save(f, H)

        path = os.path.join(PATH, 'saved_models', '{}_{}_{}_hh.npy'.format(epochs, time_steps, APPLICATION))
        with open(path, 'wb') as f:
            np.save(f, H_H)

        H_H = H_H[-1]

        ##state = H_H.reshape((1, 28, 1, 28, 1)).max(4).max(2)
        #state = predict(W, small_image)
        #state = H_H.reshape((1, 22, 1, 22, 1)).max(4).max(2)
        #state = state.reshape((1, 14, 1, 14, 1)).max(4).max(2)

        ##state = np.squeeze(state, axis=0)
        ##path = os.path.join(PATH, 'plots', '{}_{}_{}_h.png'.format(epochs, time_steps, APPLICATION))
        ##plt.clf()
       ## plot_utils.save_image(state, '', '', "Associative activation (H)", path)
        ##plt.show()
        x = 1
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
    y = np.zeros((0), dtype=int)
    train_indexes= []

    for i in range(Nc):
        startIdx = len(data)
        train_indexes += list(range(startIdx, startIdx + 1))
        data=np.concatenate((data, mat['train'+str(i)]), axis=0)
        y=np.concatenate((y, np.full((mat['train'+str(i)].shape[0]), i)), axis=0)
    

    enc = preprocessing.OneHotEncoder()
    enc.fit(y.reshape(-1, 1))

    y = enc.transform(y.reshape(-1, 1)).toarray()
    print(y.shape)
    data = data[train_indexes]
    y = y[train_indexes]

    data_size = data.shape[0]

    data = data/255.0
    data_input_size = 28
    data_output_size = 14
    bin_size = data_input_size // data_output_size

    #2(0), 20000 (3), 55000 (9)
    network = AssociativeNetwork(args.no_of_units, args.time_steps)
    APPLICATION = 2
    train(network, data, y, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold, data_output_size, bin_size, APPLICATION)
    x = 1
    #network = AssociativeNetwork(args.no_of_units, args.time_steps)
    #APPLICATION = 20000
    #train(network, data, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold, data_output_size, bin_size, APPLICATION)

    #network = AssociativeNetwork(args.no_of_units, args.time_steps)
    #APPLICATION = 55000
    #train(network, data, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold, data_output_size, bin_size, APPLICATION)

if __name__ == '__main__':
    main()
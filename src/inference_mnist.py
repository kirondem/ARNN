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
from models.associative_inference import AssociativeInference
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
    parser.add_argument('--time_steps', default=1, type=int)
    parser.add_argument('--no_of_units', default=784, type=int, help='Number of units in the associative network')
    parser.add_argument('--no_of_input_units', default=784, type=int, help='Number of input units')
    
    return parser.parse_args()

def inference(network, data, data_size, batch_size, epochs, time_steps, lr, decay_threshold, APPLICATION):

    start_time = time.time()

    logging.info("Start training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))

    for epoch in range(epochs):

        # decay learning rate
        learning_rate = lr * (1 - epoch / epochs)

        logging.info('Epoch {}, lr {}'.format( epoch, learning_rate))

        # Randomise the data
        #data = data[np.random.permutation(data_size),:] # Randomise the data

        data = data[APPLICATION] #2, 20000, 55000
        data = data.reshape((1, data.shape[0]))
        
        # Iterate over data.
        for i in range(data_size//batch_size):
            
            input = data[i*batch_size: (i+1)*batch_size]
            input = input.flatten()

            W, H, H_H, avg_w_list, active_units_list, assoc_active_units_list = network.learn(input, time_steps, learning_rate, decay_threshold)


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
    for i in range(Nc):
        data=np.concatenate((data, mat['train'+str(i)]), axis=0)
        y=np.concatenate((y, np.full((mat['train'+str(i)].shape[0]), i)), axis=0)
    
    data = data/255.0

    #2(0), 20000 (3), 55000 (9)
    network = AssociativeInference(args.no_of_units, args.time_steps)
    APPLICATION = 2
    inference(network, data, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold, APPLICATION)

if __name__ == '__main__':
    main()
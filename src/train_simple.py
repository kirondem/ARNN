import logging
import argparse
import os
import scipy.io
import numpy as np
from lib import enums, constants
from configparser import ConfigParser
from matplotlib import pyplot as plt
from models.associative_network import AssociativeNetwork

LOG_LEVEL = logging.getLevelName(constants.lOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)
PATH  =  os.path.dirname(os.path.abspath(__file__))

config_parser = ConfigParser()
config_parser.read('configuration.ini')

epochs = 1 # number of epochs
lr = float(config_parser.get('Network_Config', 'learning_rate'))
lr = 0.001
decay_threshold = 0.01

def get_args_parser():
    parser = argparse.ArgumentParser('Train associative network', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--lr', default=lr, type=float)
    parser.add_argument('--time_steps', default=3, type=int)

    return parser.parse_args()

def plot_compare(values1, values2, bins=10, range=None):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111) 
    ax.hist(values1.ravel(), alpha=0.5, bins=bins, range=range, color= 'b', label='1')
    ax.hist(values2.ravel(), alpha=0.5, bins=bins, range=range, color= 'r', label='2')
    ax.legend(loc='upper right', prop={'size':10})
    plt.show()

def main():
    args = get_args_parser()

    data = np.array([0.1, 0.1, 0.9])

    network = AssociativeNetwork(args.time_steps)

    for epoch in range(epochs):

        # decay learning rate
        learning_rate = lr * (1 - epoch / epochs)
        W, H, H_H = network.learn(data, args.time_steps, learning_rate, decay_threshold)

    H = H[-2]
    H_H = H_H[-2]
    
    #plot_compare(H, H_H)

    h = [data, H, H_H]
    l = ['X', 'H', 'H_H']
    x = [str(t) for t in range(len(H))]

    #for i in range(len(h)):
    #    plt.plot(x, h[i], label=l[i])

    #plt.xlabel('units')
    #plt.legend()
    #plt.show()

    weights = []
    for t in range(len(W)):
        for w_from in range(len(W[0])):
            unit_weights = []
            for w_to in range(len(W[0])):
                unit_weights.append(W[w_from][w_to])

            weights.append(unit_weights)

    #print(weights)
    #print(W[-1])
    

    x = [str(t) for t in range(len(W))]
    w0_0, w0_1, w0_2, w1_0, w1_1, w1_2, w2_0, w2_1, w2_2 = [], [], [], [], [], [], [], [], []

    from_unit = 0
    to_unit = 0

    for t in range(len(W)):

        for w_from in range(len(W[0])):

             for w_to in range(len(W[0])):
                    pass

              
    for t in range(len(W)):
        
        w0_1.append(W[t][0][1])
        w0_2.append(W[t][0][2])

        w1_0.append(W[t][1][0])
        w1_2.append(W[t][1][2])

        w2_0.append(W[t][2][0])
        w2_1.append(W[t][2][1])
        
    
    w = [w0_1, w0_2, w1_0, w1_2, w2_0, w2_1]

    labels = [ 'w0_1', 'w0_2', 'w1_0',  'w1_2', 'w2_0', 'w2_1', ]

    for i in range(len(w)):
        
        plt.plot(x, w[i], label=labels[i])
        # plt.plot(x, w[i], label='W_'+str(from_unit)+ '_' + str(to_unit))

    plt.legend()

    #plt.plot([i for i in range(len(weights))], weights)
    plt.show()
    t= 1


if __name__ == '__main__':
    main()
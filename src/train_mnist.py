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
import seaborn as sns
from lib.utils import dynamic_lambda

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
            
            input = data[i*batch_size: (i+1)*batch_size]
            input = input.flatten()
            
            W, Wt_LAST, H, H_H = network.learn(input, time_steps, learning_rate, decay_threshold)

            # Reseting the network timesteps
            network.reset(time_steps, Wt_LAST)

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
    

def associate(network1, network2, data1, data2, data_size, batch_size, epochs, time_steps, lr, decay_threshold):

    start_time = time.time()
    logging.info("Start associate {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))

    no_of_output_units = network2.no_of_units
    no_of_units = network1.no_of_units

    W_ASSOC = np.zeros((no_of_output_units, no_of_units)) 

    for epoch in range(epochs):

        # decay learning rate
        learning_rate = lr * (1 - epoch / epochs)
        logging.info('Epoch {}, lr {}'.format( epoch, learning_rate))
        #data = data.reshape((1, data.shape[0]))
        
        # Iterate over data.
        for i in range(data_size//batch_size):
            
            input = data1[i*batch_size: (i+1)*batch_size].flatten()
            W, Wt_LAST1, H, H_H1 = network1.learn(input, time_steps, learning_rate, decay_threshold)
            network1.reset(time_steps, Wt_LAST1)

            input = data2[i*batch_size: (i+1)*batch_size].flatten()
            W, Wt_LAST2, H, H_H2 = network2.learn(input, time_steps, learning_rate, decay_threshold)
            network2.reset(time_steps, Wt_LAST2)

            W_ASSOC = associate_inputs(W_ASSOC, learning_rate, H_H1[-1], H_H2[-1])

    
    end_time = time.time()

    logging.info("End associate {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logging.info("associate took {0:.1f}".format(end_time-start_time))

    return W_ASSOC

def main():

    args = get_args_parser()

    Nc = 10
    N = 784
    data_size = 1 # 60000
    batch_size = 1
    decay_threshold = 0.1

    mat = scipy.io.loadmat(os.path.join(PATH, 'data', 'mnist_all.mat'))
    data_mnist = np.zeros((0, N))
    y = np.zeros((0), dtype=int)
    train_indexes= []

    for i in range(Nc):
        startIdx = len(data_mnist)
        train_indexes += list(range(startIdx, startIdx + 1))
        data_mnist=np.concatenate((data_mnist, mat['train'+str(i)]), axis=0)
        y=np.concatenate((y, np.full((mat['train'+str(i)].shape[0]), i)), axis=0)

    #Read in fashion mnist data
    X_fashion_mnist_data = np.zeros((0, N))
    y_fashion_mnist_data = np.zeros((0, N))

    path = os.path.join(PATH, 'data', 'fashion-mnist','images-idx3-ubyte.npy')
    with open(path, 'rb') as f:
        X_fashion_mnist_data = np.load(f)
    
    path = os.path.join(PATH, 'data', 'fashion-mnist','labels-idx1-ubyte.npy')
    with open(path, 'rb') as f:
        y_fashion_mnist_data = np.load(f)

    #TODO: Remove hard coded list of indexes
    train_indexes = [0, 5923, 12665, 25623, 30596, 50596]
    train_indexes = [0]
    # mnist labels 0, 1, 2, 4, 5, 8     #Fashion labels 9, 3, 4, 5, 8, 4


    train_even_indexes = [12665, 25623, 39000, 50596] # 2, 4, 6, 8
    train_odd_indexes = [5923, 20000, 30596, 46000] # 1, 3, 5, 7
    sandles_indexes = [9, 12, 13, 30 ]
    handbag_indexes = [3, 5, 8, 10]

    ##data_mnist = data_mnist[train_indexes]
    ##data_size = data_mnist.shape[0]
    ##data_mnist = data_mnist/255.0

    data_mnist_even = data_mnist[train_even_indexes]
    data_mnist_even = data_mnist_even.shape[0]
    data_mnist_even = data_mnist_even/255.0

    data_mnist_odd = data_mnist[train_odd_indexes]
    data_mnist_odd = data_mnist_odd.shape[0]
    data_mnist_odd = data_mnist_odd/255.0

    X_fashion_mnist_data = X_fashion_mnist_data[sandles_indexes]
    X_fashion_mnist_data = X_fashion_mnist_data/255.0

    #Nothing image
    nothing_image = np.zeros((28,28))
    #img = np.zeros([28,28],dtype=np.uint8)
    

    network_D = AssociativeNetwork(args.no_of_units, args.time_steps)
    W, Wt_LAST, H, H_H = train(network_D, data_mnist, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
    Wt_LAST_D = Wt_LAST.copy()

    network_F = AssociativeNetwork(args.no_of_units, args.time_steps)
    W, Wt_LAST, H, H_H = train(network_F, X_fashion_mnist_data, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
    Wt_LAST_F = Wt_LAST.copy()

    network_D.reset(args.time_steps, Wt_LAST_D)
    network_F.reset(args.time_steps, Wt_LAST_F)
    associate(network_D, network_F, data_mnist, X_fashion_mnist_data, data_size, batch_size, args.epochs, args.time_steps, args.lr, decay_threshold)
    x = 1

if __name__ == '__main__':
    main()
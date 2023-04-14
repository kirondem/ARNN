import logging
import os
from matplotlib import pyplot as plt
import numpy as np
import tqdm
from statistics import mean
from base import Base
from lib import utils, enums, constants, activation_functions
from learning import Learning
from lib.plot_utils import save_plot_H, save_plot

PATH  =  os.path.dirname(os.path.abspath(__file__))
class AssociativeNetwork(Base):

    def __init__(self, no_of_units, time_steps):
        super().__init__()
        self.no_of_units = no_of_units
        self.time_steps = time_steps
        self.learning = Learning(no_of_units)

        self.W = [0] * time_steps   # Weights matrix       
        self.H = [0] * time_steps   # Units vector
        self.H_H = [0] * time_steps # Units associatively activated vector
        self.DECAYED_IDXS = [0] * time_steps # decayed_activations_idxs
        self.DECAYED_ACTIVATIONS = [0] * time_steps # decayed_activations

        # Set the initial time step values
        self.W[0] = np.zeros((no_of_units, no_of_units))
        self.W_OUT = np.zeros((10, no_of_units))   # Head Weights matrix
        self.H[0] = np.zeros(no_of_units)
        self.H_H[0] = []
        self.DECAYED_IDXS[0] = []
        self.DECAYED_ACTIVATIONS[0] = np.zeros(no_of_units)
    
    def init_weights(self, W):
        self.W[0] = W

    def reset(self, time_steps):
        self.time_steps = time_steps
        self.W = [0] * self.time_steps   # Weights matrix   
        self.H = [0] * self.time_steps   # Units vector
        self.H_H = [0] * self.time_steps # Units associatively activated vector
        self.DECAYED_IDXS = [0] * self.time_steps # decayed_activations_idxs
        self.DECAYED_ACTIVATIONS = [0] * self.time_steps # decayed_activations

        self.W[0] = np.zeros((self.no_of_units, self.no_of_units))
        self.H[0] = np.zeros(self.no_of_units)
        self.H_H[0] = []
        self.DECAYED_IDXS[0] = []
        self.DECAYED_ACTIVATIONS[0] = np.zeros(self.no_of_units)

    def predict(self, x, y, time_steps, learning_rate, decay_threshold):

        for t in range(time_steps):

            logging.info("Timestep (predict) {}/{}".format(t, time_steps))

            # 1) Direct activation of the excitory units 
            directly_activated_units_idxs = self.learning.direct_activation_of_units(x, self.H[t])

            # 2) Associative activation
            H_H = self.learning.associative_activations(self.W[t], self.DECAYED_IDXS[t], self.DECAYED_ACTIVATIONS[t])
            self.H_H[t] = H_H.copy()

            # 3) Learning rule (Update the weights)
            W = self.learning.update_weights(self.W[t], self.H[t], learning_rate, directly_activated_units_idxs, self.DECAYED_IDXS[t], self.DECAYED_ACTIVATIONS[t])

            # 4) Decay the activations
            decayed_activations, decayed_activations_idxs = utils.decay_activation_g(self.H[t], t + 1, decay_threshold, time_steps)

            if t < time_steps - 1:
                self.W[t + 1] = W.copy()
                self.DECAYED_IDXS[t + 1]   = decayed_activations_idxs.copy()
                decayed_activations = decayed_activations.copy()
                self.DECAYED_ACTIVATIONS[t + 1] = decayed_activations
                self.H[t + 1] = decayed_activations
    
        H_OUT = self.H_H[-1]
        Zh = np.dot(H_OUT.reshape(1, -1), np.transpose(self.W_OUT))
        Zh = Zh.flatten()
        max = np.argmax(Zh)
        
        print("Actual: ", np.argmax(y))
        print('predicted:', max)

        return W

    def learn(self, x, time_steps, learning_rate, decay_threshold):

        avg_w_list = []

        for t in range(time_steps):

            logging.info("Timestep {}/{}".format(t + 1, time_steps))

            # 1) Direct activation of the excitory units 
            directly_activated_units_idxs = self.learning.direct_activation_of_units_optimised(x, self.H[t])

            # 2) Associative activation
            H_H = self.learning.associative_activations(self.W[t], self.DECAYED_IDXS[t], self.DECAYED_ACTIVATIONS[t])
            self.H_H[t] = H_H.copy()

            # 3) Learning rule (Update the weights)
            W = self.learning.update_weights(t, self.W[t], self.H[t], self.H_H[t], learning_rate, directly_activated_units_idxs, self.DECAYED_IDXS[t], self.DECAYED_ACTIVATIONS[t])

            # 4) Decay the activations
            decayed_activations, decayed_activations_idxs = utils.decay_activation_g(self.H[t], t + 1, decay_threshold, time_steps)

            if t < time_steps - 1:
                self.W[t + 1] = W.copy()
                self.DECAYED_IDXS[t + 1]   = decayed_activations_idxs.copy()
                decayed_activations = decayed_activations.copy()
                self.DECAYED_ACTIVATIONS[t + 1] = decayed_activations
                self.H[t + 1] = decayed_activations

            avg_w_list.append(mean(W.mean(axis=0)))

        #save_plot_H(self.H, "Activation", "Time steps", "H Activations", os.path.join('plots', 'H'), time_steps)
        
        #save_plot_H(self.H_H, "Activation", "Time steps", "H_H Activations", os.path.join('plots', 'H_H'), time_steps)

        x_ticks = [str(i + 1) for i in range(len(avg_w_list))]
        
        #save_plot(avg_w_list, "Avg weight", "Time steps", "Average weight", os.path.join('plots', "avg_w_{}.png".format(time_steps)), x_ticks)


        return self.W, W, self.H, self.H_H



        
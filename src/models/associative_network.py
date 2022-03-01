import logging
import os
from matplotlib import pyplot as plt
import numpy as np
import tqdm
from statistics import mean
from base import Base
from lib import utils
from learning import Learning
from lib.plot_utils import save_plot_H, save_plot

class AssociativeNetwork(Base):

    def __init__(self, no_of_units, time_steps):
        super().__init__()
        self.learning = Learning(no_of_units)

        self.W = [0] * time_steps   # Weights matrix
        self.H = [0] * time_steps   # Units vector
        self.H_H = [0] * time_steps # Units associatively activated vector
        self.DECAYED_IDXS = [0] * time_steps # decayed_activations_idxs
        self.DECAYED_ACTIVATIONS = [0] * time_steps # decayed_activations

        # Set the initial time step values
        self.W[0] = np.zeros((no_of_units, no_of_units))
        self.H[0] = np.zeros(no_of_units)
        self.H_H[0] = []
        self.DECAYED_IDXS[0] = []
        self.DECAYED_ACTIVATIONS[0] = np.zeros(no_of_units) 

    def learn(self, x, time_steps, learning_rate, decay_threshold):

        # Collect the network activity at all time steps
        sum_x_list = []
        sum_h_list = []
        sum_hh_list = []
        sum_w_list = []
        avg_w_list = []

        for t in tqdm.tqdm(range(time_steps)):

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

            sum_x = sum(x)
            sum_h = sum(self.H[t])
            sum_hh = sum(self.H_H[t])
            sum_w = sum(map(sum, W))

            sum_x_list.append(sum_x)
            sum_h_list.append(sum_h)
            sum_hh_list.append(sum_hh)
            sum_w_list.append(sum_w)
            a = mean(W.mean(axis=0))
            avg_w_list.append(a)


        save_plot_H(self.H, "Activation", "Time steps", "H Activations", os.path.join('plots', 'H'), time_steps)
        
        save_plot_H(self.H_H, "Activation", "Time steps", "H_H Activations", os.path.join('plots', 'H_H'), time_steps)

        x_ticks = [str(i + 1) for i in range(len(avg_w_list))]
        
        save_plot(avg_w_list, "Avg weight", "Time steps", "Average weight", os.path.join('plots', "avg_w_{}.png".format(time_steps)), x_ticks)

        return self.W, self.H, self.H_H



        
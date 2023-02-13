import logging
import time
import numpy as np
import numpy as np
import random
from base import Base
from scipy.stats import norm
from lib import enums, constants
from lib.activation_functions import htan, relu
import matplotlib
import matplotlib.pylab as plt

from lib.utils import dynamic_lambda, lambda_US_magnitude

LOG_LEVEL = logging.getLevelName(constants.lOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)
#matplotlib.use('TkAgg',force=True)

class Learning(Base):

    def __init__(self, no_of_units):
        super().__init__()
        self.lambda_max = super().lambda_max
        self.no_of_units = no_of_units

    def direct_activation_of_units(self, x, H_t):
        activated_units_idxs = np.where(x > H_t)[0]
        np.put(H_t, activated_units_idxs, x[activated_units_idxs])
        return activated_units_idxs

    def direct_activation_of_units_optimised(self, x, H_t):
        random_idxs = np.array(random.sample(range(len(H_t)), len(x)))
        x_idxs = np.where(x > H_t[random_idxs])[0]
        np.put(H_t, random_idxs[x_idxs], x[x_idxs])
        return random_idxs[x_idxs]

    def direct_activation_of_units_randomly(self, x, H_t):

        """ Randomly activate units, if they are not already activated by decay values max"""
        #TODO: Check that same indexes havent been selected
        random_idxs = random.sample(range(len(H_t)), len(x))      
        for i, random_idx in enumerate(random_idxs):  
            if (x[i] > H_t[random_idx]):
                H_t[random_idx] = x[i]
        return random_idxs

    def associative_activations(self, W, decayed_activations_idxs, decayed_activations):
        
        H_H = np.zeros(self.no_of_units)

        for decayed_unit_idx in decayed_activations_idxs:

            total_activation = 0
            for idx in decayed_activations_idxs:
                
                if idx != decayed_unit_idx:
                    total_activation +=  W[idx][decayed_unit_idx] * decayed_activations[idx]
            
            H_H[decayed_unit_idx] = total_activation
        
        # Activation function
        ##H_H = relu(H_H)

        return H_H

    def update_weights(self, t, W, h, h_h, learning_rate, directly_activated_units_idxs, decayed_activations_idxs, decayed_activations):
        
        W = W.copy()
 
        unique_decayed_activations_idxs = [i for i in decayed_activations_idxs if i not in directly_activated_units_idxs]

        #print(decayed_activations)
        count = 0
        for from_idx in directly_activated_units_idxs:
            #logging.info("{} of {}".format(count, len(directly_activated_units_idxs)))

            if h[from_idx] != 0:

                total_direct_activations = 0
                total_decayed_activations = 0
                total_associative_activations = 0
                
                # Look over other directly actived units
                for to_idx in directly_activated_units_idxs:
                    if from_idx != to_idx:
                        
                        # 1) Calculate total direct activations
                        
                        total_direct_activations = sum([W[idx][to_idx] * h[idx] for idx in directly_activated_units_idxs])
                        
                        #if t > 0: 
                        # 2) Calculate total decayed activations
                        total_decayed_activations = sum([W[idx][to_idx] * decayed_activations[idx] for idx in unique_decayed_activations_idxs])
                        
                        # 3) Calculate total associative activations
                        total_associative_activations = sum([W[idx][to_idx] * h_h[idx] for idx in decayed_activations_idxs])
                        
                        #print(total_decayed_activations, ':', total_associative_activations)
                        
                        v_total = total_direct_activations + total_decayed_activations + total_associative_activations

                        h_to = h[to_idx]
                        h_from = h[from_idx]
                        
                        # 3) Calculate maximum conditioning possible for the US
                        to_lambda_max = dynamic_lambda(h_from, h_to)
                        #to_lambda_max = lambda_US_magnitude(h_to)

                        d_w = learning_rate * h_from * ((to_lambda_max * h_to) - v_total)

                        # print("lambda max:", to_lambda_max)
                        # print("v_total:", v_total)
                        # print("d_w:", d_w)

                        W[from_idx][to_idx] = W[from_idx][to_idx] + d_w
            count += 1

        #logging.info("Time taken to update weights: {}".format(end - start))
        

        """ Prune the smallest weights induced by plasticity mechanisms; Apply lower cutoff weight"""
        #wee_t = self.init.prune_small_weights(wee_t, cutoff_weights[0])

        """Check and set all weights < upper cutoff weight """
        #wee_t = self.init.set_max_cutoff_weight(wee_t, cutoff_weights[1])

        return W




import logging
import queue
import threading
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
import math

from lib.utils import dynamic_lambda, lambda_US_magnitude, lambda_set_to_1, prune_small_weights, set_max_cutoff_weight

LOG_LEVEL = logging.getLevelName(constants.lOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)
#matplotlib.use('TkAgg',force=True)

class Learning(Base):

    def __init__(self, no_of_units):
        super().__init__()
        self.lambda_max = super().lambda_max
        self.no_of_units = no_of_units

    def direct_activation_of_units(self, x, H_t):
        x_idxs = np.where(x > H_t)[0]
        np.put(H_t, x_idxs, x[x_idxs])
        return x_idxs, H_t

    def direct_activation_of_units_optimised(self, x, H_t):
        # random_idxs = np.array(random.sample(range(len(H_t)), len(x))) 
        random_idxs = np.random.choice(len(H_t), len(x), replace=False)
        x_idxs = np.where(x > H_t[random_idxs])[0]
        np.put(H_t, random_idxs[x_idxs], x[x_idxs])
        return random_idxs[x_idxs], H_t

    def direct_activation_of_units_randomly(self, x, H_t):

        """ Randomly activate units, if they are not already activated by decay values max"""
        #TODO: Check that same indexes havent been selected
        random_idxs = random.sample(range(len(H_t)), len(x))      
        for i, random_idx in enumerate(random_idxs):  
            if (x[i] > H_t[random_idx]):
                H_t[random_idx] = x[i]
        return random_idxs

    def associative_activations_optimised(self, W, decayed_activations_idxs, decayed_activations):
        # Use numpy to initialize the array
        H_H = np.zeros(self.no_of_units)

        for decayed_unit_idx in decayed_activations_idxs:

            # Use numpy to remove the current unit from the list of decayed units
            other_idxs = np.delete(decayed_activations_idxs, np.where(decayed_activations_idxs == decayed_unit_idx))

            # Use numpy to perform element-wise operations on the arrays
            H_H[decayed_unit_idx] = np.sum(W[other_idxs, decayed_unit_idx] * decayed_activations[other_idxs])


        return H_H

    def associative_activations(self, W, decayed_activations_idxs, decayed_activations):
        
        H_H = np.zeros(self.no_of_units)

        for decayed_unit_idx in decayed_activations_idxs:

            #total_activation = 0
            #for idx in decayed_activations_idxs:
                
                #if idx != decayed_unit_idx:
                    #total_activation +=  W[idx][decayed_unit_idx] * decayed_activations[idx]
            
            H_H[decayed_unit_idx] = sum(W[idx][decayed_unit_idx] * decayed_activations[idx] for idx in decayed_activations_idxs if idx != decayed_unit_idx)

            #H_H[decayed_unit_idx] = total_activation
        
        # H_H1 = [sum(W[idx][decayed_unit_idx] * decayed_activations[idx] for idx in decayed_activations_idxs if idx != decayed_unit_idx) for decayed_unit_idx in decayed_activations_idxs]

        # Activation function
        #H_H = relu(H_H)

        return H_H

    
    def total_activations(W, to_idx, h, idxs, out_queue1):
        total = sum([W[idx][to_idx] * h[idx] for idx in idxs])
        out_queue1.put(total)
        #print("All done in the new thread:", threading.current_thread().name)

    def update_weights_optimised(self, t, W, h, h_h, learning_rate, directly_activated_units_idxs, decayed_activations_idxs, decayed_activations):

        unique_decayed_activations_idxs = [i for i in decayed_activations_idxs if i not in directly_activated_units_idxs]
        v_total = 0
        d_w = 0

        for from_idx in directly_activated_units_idxs:

            if h[from_idx] != 0:

                # Use list comprehension to calculate the total direct, decayed, and associative activations
                total_direct_activations = sum([W[idx][to_idx] * h[idx] for idx in directly_activated_units_idxs for to_idx in directly_activated_units_idxs if from_idx != to_idx])
                total_decayed_activations = sum([W[idx][to_idx] * decayed_activations[idx] for idx in unique_decayed_activations_idxs for to_idx in directly_activated_units_idxs])
                total_associative_activations = sum([W[idx][to_idx] * h_h[idx] for idx in decayed_activations_idxs for to_idx in directly_activated_units_idxs])
                v_total = total_direct_activations + total_decayed_activations + total_associative_activations

                for to_idx in directly_activated_units_idxs:
                    if from_idx != to_idx:
                        h_to = h[to_idx]
                        h_from = h[from_idx]

                        to_lambda_max = dynamic_lambda(h_from, h_to)
                        d_w = learning_rate * h_from * (((to_lambda_max) - (v_total)))

                        # Use a smaller learning rate to reduce the magnitude of the weight updates
                        # d_w = d_w / 10

                        if(math.isnan(d_w)):
                            print("h_from:", h_from, " lambda max:", to_lambda_max, " v_total:", v_total, " d_w:", d_w)

                        W[from_idx][to_idx] = W[from_idx][to_idx] + d_w

            # Use numpy to apply the pruning and cutoff weights
            W = prune_small_weights(W, -1)
            W = set_max_cutoff_weight(W, 1)
        print("v_total:", v_total, "d_w:", d_w)
        return W

    def update_weights(self, t, W, h, h_h, learning_rate, directly_activated_units_idxs, decayed_activations_idxs, decayed_activations):
 
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

                        # 2) Calculate total decayed activations
                        total_decayed_activations = sum([W[idx][to_idx] * decayed_activations[idx] for idx in unique_decayed_activations_idxs])

                        # 3) Calculate total associative activations
                        total_associative_activations = sum([W[idx][to_idx] * h_h[idx] for idx in decayed_activations_idxs])
                        #print(total_decayed_activations, ':', total_associative_activations)

                        #N = len(directly_activated_units_idxs)  + len(unique_decayed_activations_idxs) + len(decayed_activations_idxs)
                        N = len(directly_activated_units_idxs)
                        norm = 1 / N

                        v_total = total_direct_activations + total_decayed_activations + total_associative_activations

                        h_to = h[to_idx]
                        h_from = h[from_idx]
                        
                        # 3) Calculate maximum conditioning possible for the US
                        
                        #to_lambda_max = lambda_US_magnitude(h_to)
                        
                        # TESTS
                        # to_lambda_max = dynamic_lambda(h_from, h_to)
                        # 1) d_w = learning_rate * h_from * ((to_lambda_max * h_to) - v_total)
                        
                        # 2) Removed * h_to
                        # to_lambda_max = dynamic_lambda(h_from, h_to)
                        #d_w = learning_rate * h_from * ((to_lambda_max) - v_total)

                        # 3) lambda_set_to_1
                        #to_lambda_max = lambda_set_to_1()
                        #d_w = learning_rate * h_from * ((to_lambda_max) - v_total)

                        # 4) 
                        #to_lambda_max = dynamic_lambda(h_from, h_to)
                        # d_w = learning_rate * h_from * (((to_lambda_max * norm) - (v_total)) **2)

                        # 5)
                        to_lambda_max = dynamic_lambda(h_from, h_to)
                        d_w = learning_rate * h_from * (((to_lambda_max) - (v_total)))

                        if(math.isnan(d_w)):
                            print("h_from:", h_from, " lambda max:", to_lambda_max, " v_total:", v_total, " d_w:", d_w)

                        W[from_idx][to_idx] = W[from_idx][to_idx] + d_w

                #print("lambda max:", to_lambda_max, "v_total:", v_total, "d_w:", d_w)

            count += 1

        #logging.info("Time taken to update weights: {}".format(end - start))
        
        """ Prune the smallest weights induced by plasticity mechanisms; Apply lower cutoff weight"""
        W = prune_small_weights(W, -1)

        """Check and set all weights < upper cutoff weight """
        #wee_t = self.init.set_max_cutoff_weight(wee_t, cutoff_weights[1])

        W = set_max_cutoff_weight(W, 1)
        
        print("v_total:", v_total, "d_w:", d_w)

        return W




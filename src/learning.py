import numpy as np
import numpy as np
import random
from base import Base
from scipy.stats import norm
from lib import enums
from lib.activation_functions import htan, relu

class Learning(Base):

    def __init__(self, no_of_units):
        super().__init__()
        self.lambda_max = super().lambda_max
        self.no_of_units = no_of_units

    def update_weights(self, W, h, learning_rate, directly_activated_units_idxs, decayed_activations_idxs, decayed_activations):
        
        #decayed_direct_activation_idxs = np.argwhere(h > 0.)
        #decayed_direct_activation_idxs= np.squeeze(decayed_direct_activation_idxs, axis=1)
 
        unique_decayed_activations_idxs = [i for i in decayed_activations_idxs if i not in directly_activated_units_idxs]

        associative_activations_idxs = []

        #print(decayed_activations)
        for from_idx in directly_activated_units_idxs:

            if h[from_idx] != 0:
                
                # Look over other directly actived units
                for to_idx in directly_activated_units_idxs:
                    if from_idx != to_idx:
                        
                        total_direct_activations = 0
                        total_decayed_activations = 0
                        total_associative_activations = 0

                        # 1) Calculate total direct activations
                        for idx in directly_activated_units_idxs:
                            if idx != to_idx and idx != from_idx and W[idx][to_idx] !=0 and h[idx] != 0:
                                total_direct_activations = total_direct_activations + W[idx][to_idx] * h[idx]
                        
                        # 2) Calculate total decayed activations
                        for idx in unique_decayed_activations_idxs:
                            if idx != to_idx and idx != from_idx and W[idx][to_idx] != 0 and decayed_activations[idx] != 0:
                                total_decayed_activations = total_decayed_activations + W[idx][to_idx] * decayed_activations[idx]

                        # 3) Calculate total associative activations
                        for idx in decayed_activations_idxs:
                            # TODO: Check if this is correct
                            total_associative_activations = total_associative_activations
                            

                        ##print('------------------')
                        
                        h_to = h[to_idx]
                        h_from = h[from_idx]

                        ##print('h_from: ', h_from)
                        ##print('h_to: ', h_to)
                        

                        # 3) Calculate maximum conditioning possible for the US
                        to_lambda_max = self.dynamic_lambda(h_from, h_to)

                        
                        v_total = total_direct_activations + total_decayed_activations + total_associative_activations
                        ##print('v_total: ', v_total)
                        v_total = htan(v_total)
                        ##print('v_total (htan): ', v_total)

                        #h_to = 1
                        #h_from = 1
                        
                        d_w = learning_rate * h_from * (to_lambda_max * h_to - (v_total))
                        
                        if 1 == 2:
                            #print('------------------')
                            print("lambda max:", to_lambda_max)
                            print("v_total:", v_total)
                            print("d_w:", d_w)
                        
                        W[from_idx][to_idx] = W[from_idx][to_idx] + d_w

        """ Prune the smallest weights induced by plasticity mechanisms; Apply lower cutoff weight"""
        #wee_t = self.init.prune_small_weights(wee_t, cutoff_weights[0])

        """Check and set all weights < upper cutoff weight """
        #wee_t = self.init.set_max_cutoff_weight(wee_t, cutoff_weights[1])

        return W

    def associative_activations(self, W, decayed_activations_idxs, decayed_activations):
        
        H_H = np.zeros(self.no_of_units)

        for decayed_unit_idx in decayed_activations_idxs:

            total_activation = 0
            for idx in decayed_activations_idxs:
                
                if idx != decayed_unit_idx:
                    total_activation +=  W[idx][decayed_unit_idx] * decayed_activations[idx]
            
            H_H[decayed_unit_idx] = total_activation
        
        # Activation function
        H_H = relu(H_H)

        return H_H

    def dynamic_lambda(self, h_from, h_to):
            
        # Lambda max rule: The maximum weight is set to the maximum weight of the network.

        lambda_max = 1 # - (h_from * h_to)

        #lambda_max = lambda_max if lambda_max > 0 else 1

        

        return lambda_max
    
    def direct_activation_of_units(self, x, H_t):

        """ Randomly activate units, if they are not already activated by decay values max"""
        random_idxs = random.sample(range(len(H_t)), len(x))      
        for i, random_idx in enumerate(random_idxs):  
            if (x[i] > H_t[random_idx]):
                H_t[random_idx] = x[i]
        return random_idxs



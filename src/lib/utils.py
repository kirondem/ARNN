import os
import numpy as np

def dynamic_lambda(h_from, h_to):
        
    # Lambda max rule: The maximum weight is set to the maximum weight of the network.

    return (h_from * h_to)
    
    # # TODO: Suppressed 
    # #lambda_max = 1 # - (h_from * h_to)

    # #lambda_max = lambda_max if lambda_max > 0 else 1
    # return lambda_max

def decay_activation(h, t, timesteps):
    h = h * (1 - t / timesteps)
    return h

def decay_activation_g(h, t, decay_threshold,  timesteps):
    d = h.copy()
    decayed_activations = h * np.exp(-(t**2 / timesteps))
    decayed_activations[decayed_activations < decay_threshold] = 0
    decayed_activations_idxs = np.where(d!=decayed_activations)
    return decayed_activations, decayed_activations_idxs[0]

def zero_sum_incoming_check(weights):
    zero_sum_incomings = np.where(np.sum(weights, axis=0) == 0.)

    if len(zero_sum_incomings[-1]) == 0:
        return weights
    else:
        for zero_sum_incoming in zero_sum_incomings[-1]:

            rand_indices = np.random.randint(40, size=2)  # 40 in sense that size of E = 200
            # given the probability of connections 0.2
            rand_values = np.random.uniform(0.0, 0.1, 2)

            for i, idx in enumerate(rand_indices):
                weights[:, zero_sum_incoming][idx] = rand_values[i]

    return weights

def normalize_weight_matrix(weight_matrix):

    # Applied only while initializing the weight. During simulation, Synaptic scaling applied on weight matrices

    """ Normalize the weights in the matrix such that incoming connections to a neuron sum up to 1

    Args:
        weight_matrix(array) -- Incoming Weights from W_ee or W_ei or W_ie

    Returns:
        weight_matrix(array) -- Normalized weight matrix"""

    normalized_weight_matrix = weight_matrix / np.sum(weight_matrix, axis=0)

    return normalized_weight_matrix

def generate_white_gaussian_noise(mu, sigma, size):
        """Generates white gaussian noise with mean mu, standard deviation sigma and the noise length equals t """

        noise = np.random.normal(mu, sigma, size)

        return np.expand_dims(noise, 1)


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def save_parameters(PATH, epochs, time_steps, W, W_OUT, H, H_H):

    path = os.path.join(PATH, 'saved_models', '{}_{}_{}_w.npy'.format(epochs, time_steps))
    with open(path, 'wb') as f:
        np.save(f, W)

    path = os.path.join(PATH, 'saved_models', '{}_{}_{}_W_OUT.npy'.format(epochs, time_steps))
    with open(path, 'wb') as f:
        np.save(f, W_OUT)

    path = os.path.join(PATH, 'saved_models', '{}_{}_{}_h.npy'.format(epochs, time_steps))
    with open(path, 'wb') as f:
        np.save(f, H)

    path = os.path.join(PATH, 'saved_models', '{}_{}_{}_hh.npy'.format(epochs, time_steps))
    with open(path, 'wb') as f:
        np.save(f, H_H)

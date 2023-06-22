import os
import numpy as np

def dynamic_lambda(h_from, h_to):
        
    # Lambda max rule: The maximum weight is set to the maximum weight of the network.

    return (h_from * h_to)
    
    # # TODO: Suppressed 
    # #lambda_max = 1 # - (h_from * h_to)

    # #lambda_max = lambda_max if lambda_max > 0 else 1
    # return lambda_max

def lambda_US_magnitude(h_to):
    # Determined by the magnitude of the US
    return (h_to)

def lambda_set_to_1():
    # Determined by the magnitude of the US
    return (1)

def magnitude(v):
    return np.sqrt(np.sum(np.square(v)))

def normalize(v):
    return v / magnitude(v)

def sum_vector(x): 
    return sum(x)

def decay_activation(h, t, timesteps):
    h = h * (1 - t / timesteps)
    return h

def decay_activation_g(h, t, decay_threshold,  timesteps):
    decayed_activations = h * np.exp(-(t**2 / timesteps))
    decayed_activations[decayed_activations < decay_threshold] = 0
    decayed_activations_idxs = np.where(h != decayed_activations)
    return decayed_activations, decayed_activations_idxs[0]

def transform_inputs(x):
    return normalize_weight_matrix(x)

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

    normalized_weight_matrix = weight_matrix / np.sum(weight_matrix, axis=1)

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

    
def prune_small_weights(weights: np.array, cutoff_weight: float):
    """Prune the connections with negative connection strength. The weights less than cutoff_weight set to 0

    Args:
        weights (np.array): Synaptic strengths

        cutoff_weight (float): Lower weight threshold

    Returns:
        array: Connections weights with values less than cutoff_weight set to 0
    """

    weights[weights <= cutoff_weight] = cutoff_weight

    return weights


def set_max_cutoff_weight(weights: np.array, cutoff_weight: float):
    """Set cutoff limit for the values in given array

    Args:
        weights (np.array): Synaptic strengths

        cutoff_weight (float): Higher weight threshold

    Returns:
        array: Connections weights with values greater than cutoff_weight set to 1
    """

    weights[weights > cutoff_weight] = cutoff_weight

    return 

def save_weights(Path, name, data, trials, epochs, time_steps):
    path = os.path.join(Path, 'saved_weights', '{}_{}_{}_{}.npy'.format(trials, epochs, time_steps, name))
    with open(path, 'wb') as f:
        np.save(f, data)

def load_weights(Path, name, trials, epochs, time_steps, network_type):
    path = os.path.join(Path, 'saved_weights', '{}_{}_{}_{}_{}.npy'.format(trials, epochs, time_steps, name, network_type))
    with open(path, 'rb') as f:
        data = np.load(f)
    return data


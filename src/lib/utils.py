import numpy as np

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


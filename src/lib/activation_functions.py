import numpy as np

def relu(x):
    """Rectified Linear Unit"""

    return np.maximum(x, 0)

# Hyperbolic Tangent (htan) Activation Function
def htan(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def heaviside_step_activation(tot_incoming_drive, threshold):
    """Heaviside step function"""

    heaviside_step = [0] * len(tot_incoming_drive)
    for t in range(len(tot_incoming_drive)):
        heaviside_step[t] = 0.0 if tot_incoming_drive[t] < threshold[t] else 1.0

    return heaviside_step

#Sigmoid funstion
def sigmoid(x):
    return 1/(np.exp(-x)+1) 
    
def softmax(x):
    exponents=np.exp(x)
    return exponents/np.sum(exponents)
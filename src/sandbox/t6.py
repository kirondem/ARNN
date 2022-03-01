import numpy as np

def htan(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

print(htan(10.0))
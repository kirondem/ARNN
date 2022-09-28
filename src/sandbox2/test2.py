import random
import numpy as np



train_indexes = random.sample(range(0, 60000), 10)
print(train_indexes)

# A Softmax function takes in a vector as input and spits out a vector of same size having elements that sum up to 1. Every element in the output vector is between 0 and 1, and thus these values can be interpreted as probabilities.

def softmax(x):
    exponents=np.exp(x)
    return exponents/np.sum(exponents)

out=np.array([1,2,3])
print(softmax(out))
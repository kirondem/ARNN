import math
import numpy as np

def sum_vector(x): 
    return sum(x)

def mag(x): 
    return math.sqrt(sum(i**2 for i in x))

x = np.array([2, 2])

m = np.sqrt(x.dot(x))

m = np.linalg.norm(x)

m = sum_vector(x)

#m = mag(x)

print(m)
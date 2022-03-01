import numpy as np

A = np.array([1, 0.2, 0.5])

B = np.array([0.2])

A = [i for i in A if i not in B]

print(A)
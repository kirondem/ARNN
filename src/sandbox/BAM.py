import numpy as np

a1 = np.array([1, 0, 1, 0, 1, 0])
b1 = np.array([1, 1, 1, 0, 0, 0])

x = np.array([[1, 0, 1, 0, 1, 0], [1, 1, 1, 0, 0, 0]])
y = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])

#convert a binary matrix to a bipolar one
x1 = x[np.isclose(x, 0)] = -1
y1 = y[np.isclose(y, 0)] = -1

M = np.dot(x.T, y)

a = np.dot(a1, M)
b = np.dot(b1, M)

a =  1, a>=0
b =  1, b>=0
print(a)
print(b)


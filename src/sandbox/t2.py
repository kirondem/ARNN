import numpy as np

units = np.array([1, 2 ,3, 4, 5, 6, 99, 8, 100, 34])

d_a = np.array([5, 6, 2, 0, 24])

b = units < 3
print(units[b])
c = b.astype(int)

print(c)


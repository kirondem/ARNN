from numpy import dot
from numpy.linalg import norm
import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

PATH  =  os.path.dirname(os.path.abspath(__file__))
epochs = 1
time_steps = 30

APPLICATION = 2 # 20000, 55000
path = os.path.join(PATH, '..', 'saved_models', '{}_{}_{}_hh.npy'.format(epochs, time_steps, APPLICATION))
data_2 = np.load(path)
x_2 = data_2[29]

APPLICATION = 20000
path = os.path.join(PATH, '..', 'saved_models', '{}_{}_{}_hh.npy'.format(epochs, time_steps, APPLICATION))
data_20000 = np.load(path)
x_20000 = data_20000[29]

APPLICATION = 55000
path = os.path.join(PATH, '..', 'saved_models', '{}_{}_{}_hh.npy'.format(epochs, time_steps, APPLICATION))
data_55000 = np.load(path)
x_55000 = data_55000[29]

a = x_20000
b = x_55000

cos_sim = dot(a, b)/(norm(a)*norm(b))

print(cos_sim)


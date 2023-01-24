import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_Weights(data):
    plt.clf()
    plt.imshow(data, interpolation='nearest')
    plt.show()

PATH  =  os.path.dirname(os.path.abspath(__file__))
epochs = 1
time_steps = 4
trials = 8

name = 'Wt_LAST1'
file_path = os.path.join(PATH, '..', 'saved_weights', '{}_{}_{}_{}.npy'.format(trials, epochs, time_steps, name))
data = np.load(file_path)
data = data.flatten()
sns.displot(data,  kind="kde")
plt.show()

name = 'Wt_LAST2'
file_path = os.path.join(PATH, '..', 'saved_weights', '{}_{}_{}_{}.npy'.format(trials, epochs, time_steps, name))
data = np.load(file_path)
data = data.flatten()
sns.displot(data,  kind="kde")
plt.show()


name = 'W_A'
file_path = os.path.join(PATH, '..', 'saved_weights', '{}_{}_{}_{}.npy'.format(trials, epochs, time_steps, name))
data = np.load(file_path)
data = data.flatten()
sns.displot(data,  kind="kde")
plt.show()

t= 1
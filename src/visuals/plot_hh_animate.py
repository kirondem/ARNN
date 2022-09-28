import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import animation

import numpy as np
import seaborn as sns

PATH  =  os.path.dirname(os.path.abspath(__file__))
epochs = 1
time_steps = 30
APPLICATION = 55000 # 2, 55000

path = os.path.join(PATH, '..', 'saved_models', '{}_{}_{}_h.npy'.format(epochs, time_steps, APPLICATION))
data = np.load(path)


def animate(j):
    print(data[j])
    state = data[j].reshape((1, 28, 1, 28, 1)).max(4).max(2)
    state = np.squeeze(state, axis=0)

    plt.imshow(state, interpolation='nearest')
    #plt.show()


animation_1 = animation.FuncAnimation(plt.gcf(), animate, interval=10)

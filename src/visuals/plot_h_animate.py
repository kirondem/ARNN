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

path = os.path.join(PATH, '..', 'saved_models', '{}_{}_{}_w.npy'.format(epochs, time_steps, APPLICATION))
data = np.load(path)


#plt.imshow(data[29],     interpolation='nearest')
#plt.show()

def animate(j):
    c = data[j]
    plt.imshow(c, interpolation='nearest')
    plt.show()


animation_1 = animation.FuncAnimation(plt.gcf(), animate, interval=time_steps)
plt.show()

plt.show()
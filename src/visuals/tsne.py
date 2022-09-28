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

X = np.array([x_2, x_20000, x_55000])
from sklearn.manifold import TSNE
tsne = TSNE(n_components=3, random_state=0)
X_2d = tsne.fit_transform(X)

target_names = ['0', '3', '9']
y = np.array([0, 3, 9])
target_ids = range(len(target_names))

plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.show()
x= 1
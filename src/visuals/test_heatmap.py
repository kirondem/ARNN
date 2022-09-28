import os
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation

PATH  =  os.path.dirname(os.path.abspath(__file__))

path = os.path.join(PATH, 'saved_models', '1_30_2_w.npy')

W = np.load(path)

data = W[0:20, :, :]
data_to_draw = np.zeros(shape = (1, data.shape[1]))

def animate(i):
    global data_to_draw
    #data_to_draw = np.vstack((data_to_draw, data[i]))
    data_to_draw = data[i - 1]
    #if data_to_draw.shape[0] > 5:
        #data_to_draw = data_to_draw[1:]

    ax.cla()
    sns.heatmap(ax = ax, data = data_to_draw, cbar_ax = cbar_ax, cmap = "viridis",)

grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (20, 20))
ani = FuncAnimation(fig = fig, func = animate,  interval = 100) #frames = 100,

plt.show()
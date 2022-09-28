import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


np.random.seed(0)

data = np.random.rand(120, 50)
data_to_draw = np.zeros(shape = (1, data.shape[1]))

def animate(i):
    global data_to_draw
    data_to_draw = np.vstack((data_to_draw, data[i]))
    if data_to_draw.shape[0] > 5:
        data_to_draw = data_to_draw[1:]

    ax.cla()
    sns.heatmap(ax = ax, data = data_to_draw, cmap = "viridis", cbar_ax = cbar_ax, )


grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (10, 8))
ani = FuncAnimation(fig = fig, func = animate, frames = 100, interval = 100)

plt.show()
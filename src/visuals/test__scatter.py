import os
import matplotlib.pyplot as plt
import numpy as np

PATH  =  os.path.dirname(os.path.abspath(__file__))
path = os.path.join(PATH, 'saved_models', '1_2_2_w.npy')
W = np.load(path)

W = W[0:20, :, :]

# creating two array for plotting
x = np.arange(0, 784*784, 1)

rows = 1
cols = 2

fig, axs = plt.subplots(rows, cols, gridspec_kw={'hspace': 0.2,  'wspace': 0.05}, figsize=(50,50), sharey=True)
fig.suptitle('Weights through time', fontsize=16)
index = 0
#for i in range(rows):
for j in range(cols):

    y = W[index].flatten()

    ##axs[i, j].plot(x, y)
    axs[j].plot(x, y)
    #axs[i, j].set_title("W_(t=" + str(index + 1) + ")")
    index += 1

for ax in axs.flat:
    ax.set(xlabel='weights', ylabel='strength')
    #ax.grid()
    #ax.set_aspect('equal', anchor=(0, 1))

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# creating scatter plot with both negative 
# and positive axes
#plt.scatter(x, y, s=1)
  
# visualizing the plot using plt.show() function

plt.show()
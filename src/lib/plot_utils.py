import os
import numpy as np
import matplotlib.pyplot as plt

def plot_W(W):
    plt.clf()
    plt.imshow(W, interpolation='nearest')
    plt.savefig('W.png')

def plot_H(H):
    state = H.reshape((1, 28, 1, 28, 1)).max(4).max(2)
    state = np.squeeze(state, axis=0)
    plt.imshow(state, interpolation='nearest')
    plt.show()

def save_plot(data, ylabel, xlabel, title, path, xticks=None, yticks=None):
    plt.clf()
    plt.plot(list(range(0, len(data))), data, color="blue")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if xticks is not None:
        plt.xticks(ticks=range(0,len(data)) ,labels=xticks)

    plt.title(title)
    plt.savefig(path)

def save_image(data, ylabel, xlabel, title, path):
    plt.clf()
    plt.imshow(data, interpolation='nearest')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(path)

def save_plot_H(H, ylabel, xlabel, title, path, time_steps):
    plt.clf()
    

    idx = 0
    for h in range(len(H[0])):
        h_list = []
        [h_list.append(H[t][h]) for t in range(len(H))]
        x_ticks = [str(i + 1) for i in range(len(h_list))]

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(ticks=range(0,len(h_list)) ,labels=x_ticks)
        plt.title(title)
        plt.plot(h_list, label='h' + str(h))
        plt.legend()

        if(idx % 5 == 0):
            file_path = os.path.join(path, 'H_t{}_{}.png'.format(time_steps, idx))
            plt.savefig(file_path)
            plt.clf()
        
        idx += 1
    
    plt.clf()


def display_mult_images(images, titles, rows, cols):
    
    figure, ax = plt.subplots(rows,cols)  # array of axes

    for idx, img in enumerate(images):  # images is a list
        ax.ravel()[idx].imshow(img, cmap=plt.get_cmap('gray_r'))
        ax.ravel()[idx].set_title(titles[idx])

    for axis in ax.ravel():
        axis.set_axis_off()

    #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    plt.tight_layout()
    
    plt.show()
        
    
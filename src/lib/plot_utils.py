import os
import matplotlib.pyplot as plt

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
        
    
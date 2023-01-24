
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io

# ========================================================

PATH  =  os.path.dirname(os.path.abspath(__file__))

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img


def display_mult_images(images, titles, rows, cols):
    figure, ax = plt.subplots(rows,cols)  # array of axes

    for idx, img in enumerate(images):  # images is a list
        ax.ravel()[idx].imshow(img,
        cmap=plt.get_cmap('gray_r'))
        #ax.ravel()[idx].set_title(titles[idx])

    for axis in ax.ravel():
        axis.set_axis_off()

    #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    plt.tight_layout()
    
    plt.show()

# ========================================================
N = 784
Nc = 10

#Read in fashion mnist data
X_fashion_mnist_data = np.zeros((0, N))

path = os.path.join(PATH, '..', 'data', 'fashion-mnist','images-idx3-ubyte.npy')
with open(path, 'rb') as f:
    X_fashion_mnist_data = np.load(f)

handbags_indexes = [35, 57, 99, 100]
#handbags_indexes = [35]

sandles_indexes = [9, 12, 13, 30 ]
#sandles_indexes = [9]

dresses_indexes = [1064, 1077, 1093, 1108, 1115, 1120]
#dresses_indexes = [1064]

X_fashion_mnist_data_handbags = X_fashion_mnist_data[handbags_indexes]
X_fashion_mnist_data_dresses = X_fashion_mnist_data[dresses_indexes]
X_fashion_mnist_data_sandles = X_fashion_mnist_data[sandles_indexes]

nothing_image = np.zeros((28,28))

for i in range(4):
    if i % 2 == 0:
        img1 = random.sample(list(X_fashion_mnist_data_handbags), 1)[0]
        img2 = nothing_image
        s1 = concat_images(img1.reshape(28,28), img2)
        s2 = random.sample(list(X_fashion_mnist_data_sandles), 1)[0]


        display_mult_images([np.array(s2).reshape(28,28), s2.reshape(28,28)], ['', ''], 1, 2)
        
    else:
        img1 = random.sample(list(X_fashion_mnist_data_handbags), 1)[0]
        img2 = random.sample(list(X_fashion_mnist_data_dresses),1 )[0]
        s1 = concat_images(img1.reshape(28,28), img2.reshape(28,28))
        s2 = nothing_image

    

print("\nEnd demo ")
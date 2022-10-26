import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

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
        ax.ravel()[idx].imshow(img, cmap=plt.get_cmap('gray_r'))
        ax.ravel()[idx].set_title(titles[idx])

    for axis in ax.ravel():
        axis.set_axis_off()

    #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    plt.tight_layout()
    
    plt.show()

# ========================================================
N = 784
Nc = 10
mat = scipy.io.loadmat(os.path.join(PATH, '..', 'data', 'mnist_all.mat'))
data = np.zeros((0, N))
y_data = np.zeros((0), dtype=int)
train_indexes= []
for i in range(Nc):
    startIdx = len(data)
    train_indexes += list(range(startIdx, startIdx + 5))
    data=np.concatenate((data, mat['train'+str(i)]), axis=0)
    y_data=np.concatenate((y_data, np.full((mat['train'+str(i)].shape[0]), i)), axis=0)

images = []
titles = []

train_indexes = [0, 0, 5923, 20000, 30596, 46000]

train_even_indexes = [12665, 25623, 39000, 50596, 12666, 25624] # 2, 4, 6, 8
train_odd_indexes = [5923, 20000, 30596, 46000, 5924, 56000] # 1, 3, 5, 7

train_indexes = [0, 5923]
for index in train_indexes:  # 12 images
    images.append(data[index].reshape(28,28))
    titles.append(str(y_data[index]))  # title is digit label


results = concat_images(images[0], images[1])

images = [results, results]
#plt.imshow(images[0], cmap='gray')
#plt.show()


display_mult_images(images, titles, 1, 2)

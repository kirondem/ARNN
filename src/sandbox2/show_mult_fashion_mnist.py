
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import cv2

# ========================================================

PATH  =  os.path.dirname(os.path.abspath(__file__))

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

path = os.path.join(PATH, '..', 'data', 'fashion-mnist','labels-idx1-ubyte.npy')
with open(path, 'rb') as f:
    y_data = np.load(f)

path = os.path.join(PATH, '..', 'data', 'fashion-mnist','images-idx3-ubyte.npy')
with open(path, 'rb') as f:
    data = np.load(f)

images = []
titles = []

#train_indexes = random.sample(range(0, 60000), 12)
#print(train_indexes)

train_indexes = [3, 6923, 12665, 25623, 30596, 50598]


train_indexes = np.where(y_data == 1)[0]


train_indexes = train_indexes[1:5]

print(train_indexes)

#train_indexes = [35,  57,  99, 100]

# array([ 35,  57,  99, 100], dtype=int64) Dresses


for index in train_indexes:  # 12 images
    img = data[index].reshape(28,28)
    
    resized_img = cv2.resize(img, (18, 18))

    images.append(resized_img)

    titles.append(str(y_data[index]))  # title is digit label
  
display_mult_images(images, titles, 2, 2)

print("\nEnd demo ")
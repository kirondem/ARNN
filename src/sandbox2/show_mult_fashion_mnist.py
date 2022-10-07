
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io

# ========================================================

PATH  =  os.path.dirname(os.path.abspath(__file__))

def display_mult_images(images, titles, rows, cols):
    figure, ax = plt.subplots(rows,cols)  # array of axes

    for idx, img in enumerate(images):  # images is a list
        ax.ravel()[idx].imshow(img,
        cmap=plt.get_cmap('gray_r'))
        ax.ravel()[idx].set_title(titles[idx])
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

train_indexes = [0, 5923, 12665, 18623, 24754, 30596]

for index in train_indexes:  # 12 images
    images.append(data[index].reshape(28,28))
    titles.append(str(y_data[index]))  # title is digit label
  
display_mult_images(images, titles, 2, 3)

print("\nEnd demo ")
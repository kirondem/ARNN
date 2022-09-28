
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

#train_indexes = random.sample(range(0, 60000), 12)
#print(train_indexes)

for index in train_indexes:  # 12 images
    images.append(data[index].reshape(28,28))
    titles.append(str(y_data[index]))  # title is digit label
  
display_mult_images(images, titles, 10, 5)

print("\nEnd demo ")
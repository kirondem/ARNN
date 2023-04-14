
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io

# ========================================================
resized_image_dim = 50

PATH  =  os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(PATH, '..', 'data', 'shapes','star.jpg')
img = cv2.imread(image_path, 0) 
img = cv2.resize(img, (resized_image_dim, resized_image_dim))

plt.imshow(img, cmap='gray')
plt.show()

t= 1


#img1 = cv2.resize(img1, (resized_image_dim, resized_image_dim))


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from os import listdir
import random
from os.path import isfile, join

# Hopefiel Network parameters
input_width = 50
input_height = 50
number_of_nodes = input_width * input_height
learning_rate = 1

# initialise node/input array to -1, and weights array to 0
input = np.zeros((number_of_nodes))
input[True] = -1
weights = np.zeros((number_of_nodes,number_of_nodes))

#*******************************
# Main Hopefield Functions
#********************************
# Randomly fire nodes until the overall output doesn't change
# match the pattern stored in the Hopefield Net.
def calculate_output(input, weights):
    changed = True
    while changed:
        indices = list(range(len(input)))
        random.shuffle(indices)
        new_input = np.zeros((number_of_nodes))
        
        clamped_input = input.clip(min=0) # eliminate nodes with negative value, doesn't work either way
        for i in indices:
            sum = np.dot(weights[i], clamped_input)
            new_input[i] = 1 if sum >= 0 else -1
            changed = not np.allclose(input[i], new_input[i], atol=1e-3)
        input = np.array(new_input)
    
    return np.array(input)
            
# activation(W x I) = Output
# match the pattern stored in the Hopefield Net.
def calculate_output_2(input, weights):
    output = np.dot(weights,input)
    # apply threshhold
    output[output >= 0] = 1 # green in image
    output[output < 0] = -1 # purple in image
    return output


# Store the patterns in the Hopfield Network
def learn(input, weights):
    I = np.identity(number_of_nodes) # diagnol will always be 1 if input is only 1/-1
    updates = learning_rate * np.outer(input,input) - I
    updates = updates/number_of_nodes
    weights[:] = weights + updates



#*******************************
# Misc. Functions
#*******************************
# plot an array and show on the screen
def show_array(arr):
    data = arr.reshape((-1, input_width))
    plt.imshow(data) # plotting by columns
    plt.show()
    
# learn the patterns (images) placed in "patterns" folder (images of numbers 0-9)
def learn_numbers():
    for f in listdir("patterns/"):
        file = join("patterns/", f)
        if isfile(file):
            print(file)
            im = image.imread(file)
            grey = im[:,:,0] # convert to 2d array from 3channel rgb image
            grey = np.where(grey==1,-1,1) # convert white pixel to -1 and otherwise (black) to 1
            learn(grey.flatten(), weights) # convert 2d image to 1d array (2500) and store in weights

# read a test image and match the nearest pattern
def calculate_img_output(weights, image_address):
    # show the image being tested
    im = image.imread(image_address)
    grey = im[:,:,0] # convert to 2d array from 3channel rgb image
    grey = np.where(grey==1,-1,1) # convert white pixel to -1 and otherwise (black) to 1
    plt.imshow(grey) # plotting by columns
    plt.show()
    
    # retrieve the pattern using random firing
    output = calculate_output(grey.flatten(), weights)
    # show the pattern
    show_array(output)
    
    # retrieve the pattern
    output = calculate_output_2(grey.flatten(), weights)
    # show the pattern
    show_array(output)
    
    
#****************************    
# Testing code
#*****************************

    
# learn the patterns of image of number 0-9
learn_numbers()
# Try to match the partial pattern
calculate_img_output(weights, "partial/p.png")
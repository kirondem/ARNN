import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)

test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

class RescorlaWagnerNN:
    def __init__(self, input_size, alpha=0.1, beta=0.1, lambda_max=1.0):
        self.weights = torch.zeros(input_size, input_size, dtype=torch.float32) # Changed to matrix
        self.alpha = alpha
        self.beta = beta
        self.lambda_max = lambda_max

    def predict(self, inputs):
        x = torch.matmul(self.weights, inputs)
        return x

    def update_weights(self, inputs, target):
        prediction_error = target - self.predict(inputs)
        
        # Applying gradient clipping
        prediction_error = torch.clamp(prediction_error, -1.0, 1.0)

        # Reshape prediction error and inputs for outer product to get a matrix of gradients
        prediction_error = prediction_error.view(-1, 1)
        inputs = inputs.view(1, -1)
        
        self.weights += self.alpha * self.beta * prediction_error * inputs

        # Avoid nan values in weights
        #self.weights = torch.clamp(self.weights, -1.0, 1.0)


nn = RescorlaWagnerNN(input_size=28*28, lambda_max=1.0)

# Training loop
for epoch in range(100):
    for data in train_loader:
        images, _ = data
        images = images.view(-1, 28*28)
        
        for i in range(len(images)-1):
            #  associate current image with the next image in the sequence
            inputs = images[i]
            target = nn.lambda_max * images[i+1]
            
            # Update weights based on the current and next image
            nn.update_weights(inputs, target)

# Test the network with a pair of images from the test set
for data in test_loader:
    test_images, _ = data
    test_images = test_images.view(-1, 28*28)
    test_input = test_images[0]
    predicted_output = nn.predict(test_input)

    print("Predicted output: ", predicted_output.shape)
    
    # We can visualize the input and predicted output to check the associations
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(test_input.reshape((28, 28)).detach().numpy(), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Output")
    plt.imshow(predicted_output.reshape((28, 28)).detach().numpy(), cmap='gray')
    plt.show()

    # Breaking after first batch to visualize only one pair
    break
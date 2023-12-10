import numpy as np

epochs = 10

# Initialize network
input_size = 2
hidden_size = 10
output_size = 1

# Weights
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# Constants for Rescorla-Wagner
alpha = 0.1
beta = 0.5
lambda_val = 1  # Assuming B's occurrence is represented as '1'

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training
for epoch in range(epochs):
    for stimulus_A, occurrence_B in training_data:
        # Forward pass
        input_layer = np.array([stimulus_A, occurrence_B])
        hidden_layer = sigmoid(np.dot(input_layer, W1))
        output = sigmoid(np.dot(hidden_layer, W2))
        
        # Calculate error using Rescorla-Wagner rule
        error = lambda_val - output
        
        # Backward pass
        d_output = error * sigmoid_derivative(output)
        error_hidden = d_output.dot(W2.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_layer)
        
        # Update weights using Rescorla-Wagner-inspired rule
        W2 += alpha * beta * hidden_layer.T.dot(d_output)
        W1 += alpha * beta * input_layer.T.reshape(-1, 1).dot(d_hidden)

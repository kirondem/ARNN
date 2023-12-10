import numpy as np

class RWNetowrk:
    def __init__(self, input_size, alpha=0.1, beta=0.1, lambda_max=1.0):
        self.weights = np.zeros(input_size)
        self.alpha = alpha
        self.beta = beta
        self.lambda_max = lambda_max

    def predict(self, inputs):
        return np.dot(self.weights, inputs)

    def update_weights(self, inputs, target):
        prediction_error = target - self.predict(inputs)
        self.weights += self.alpha * self.beta * prediction_error * inputs

# Input data: columns represent different stimuli and rows represent different instances
# Assume two stimuli: CS (Conditioned Stimulus) and US (Unconditioned Stimulus)
input_data = np.array([
    [1, 0],  # CS present, US absent
    [1, 1],  # Both CS and US present
    [0, 1],  # CS absent, US present
    [0, 0],  # Both absent
])

# Target data: represents the desired associative strengths
target_data = np.array([
    0.0,  # No association expected as US is absent
    1.0,  # Strong association expected as both CS and US are present
    1.0,  # Strong association expected as US is present
    0.0,  # No association expected as both are absent
])

# Initializing the neural network
nn = RWNetowrk(input_size=2)

# Train neural network
for epoch in range(100):
    for inputs, target in zip(input_data, target_data):
        nn.update_weights(inputs, target)

# Testing the neural network
for inputs in input_data:
    print(nn.predict(inputs))
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neural_network(X, y):
    learning_rate = 0.1
    W1 = np.random.rand(2, 4)
    W2 = np.random.rand(4, 1)

    for epoch in range(10000):  
        layer1 = sigmoid(np.dot(X, W1))
        output = sigmoid(np.dot(layer1, W2))

        error = (y - output)

        delta2 = 2 * error * (output * (1 - output))
        delta1 = delta2.dot(W2.T) * (layer1 * (1 - layer1))

        W2 += learning_rate * layer1.T.dot(delta2)
        W1 += learning_rate * X.T.dot(delta1)

    return np.round(output).flatten()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(X.shape)
print("OR", neural_network(X, np.array([[0, 1, 1, 1]]).T))
print("AND", neural_network(X, np.array([[0, 0, 0, 1]]).T))
print("XOR", neural_network(X, np.array([[0, 1, 1, 0]]).T))
print("NAND", neural_network(X, np.array([[1, 1, 1, 0]]).T))
print("NOR", neural_network(X, np.array([[1, 0, 0, 0]]).T))
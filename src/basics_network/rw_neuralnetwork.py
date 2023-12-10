import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the Rescorla-Wagner layer
class RescorlaWagnerLayer(nn.Module):
    def __init__(self, input_size, output_size, alpha=0.1, beta=0.1, lambda_max=1):
        super(RescorlaWagnerLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
        self.alpha = alpha
        self.beta = beta
        self.lambda_max = lambda_max

    def forward(self, x):
        output = torch.mm(x, self.weights)
        
        # Implementing the Rescorla-Wagner update rule
        with torch.no_grad():
            delta_w = self.alpha * self.beta * (self.lambda_max - torch.sum(self.weights * x, dim=1, keepdim=True))
            self.weights += delta_w
        
        return output

# Define the network
class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 100)
        self.rw_layer = RescorlaWagnerLayer(100, 100)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.rw_layer(x)
        return x

# Instantiate the network, define the loss function and the optimizer
network = SimpleNetwork()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.01)

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# Load the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Training loop (only skeleton, add necessary parts for a complete training loop)
for epoch in range(2):  # Loop over the dataset multiple times
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradient buffers
        output = network(data)  # Get the network output
        loss = loss_function(output, target)  # Compute loss
        loss.backward()  # Backpropagate loss
        optimizer.step()  # Update weights

# (Add necessary parts for testing the network performance on a validation/test set)

# Print the network architecture
print(network)

network.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = network(data)
        test_loss += loss_function(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
print(f'Test loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')

# Inference function
def infer(network, image):
    network.eval()
    with torch.no_grad():
        output = network(image.unsqueeze(0))
        pred = output.argmax(dim=1)
    return pred.item()
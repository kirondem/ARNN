import torch
def activation(x):
    return 1/(1+torch.exp(-x))

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 5 random normal variables
features = torch.randn((1, 5))

# True weights for our data, random normal variables again
weights = torch.randn_like(features)

# and a true bias term
bias = torch.randn((1, 1))

#features = features.t()

product = features * weights + bias

product = torch.mm(weights, features.t() + bias

#

output = activation(product.sum())

print(output)


import torch


S = 128 # channel dim
W = 227 # width
H = 227 # height
batch_size = 10

x = torch.randn(batch_size, S, H, W)

size = 32 # patch size
stride = 32 # patch stride
patches = x.unfold(1, size, stride)
x = x.unfold(2, size, stride)
x = x.unfold(3, size, stride)
print(patches.shape)


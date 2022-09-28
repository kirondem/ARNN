import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear((16 * 5 * 5) + 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x1 = torch.ones([x.shape[0], 4], dtype=torch.float32).to(self.device)

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x2 = torch.cat((x, x1), 1)

        x = F.relu(self.fc1(x2))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

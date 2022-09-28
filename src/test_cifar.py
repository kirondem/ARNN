
import argparse
import logging
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from models.simple_cnn import SimpleCNN

LOG_LEVEL = logging.getLevelName('DEBUG')
logging.basicConfig(level=LOG_LEVEL)

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args_parser():
    parser = argparse.ArgumentParser('Solar panel production prediction', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    return parser.parse_args()

def train(model, epochs, optimizer, criterion, train_loader, test_loader, APPLICATION):

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        idx = 0
        for batch in train_loader:
            optimizer.zero_grad()

            X, y = batch
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if idx % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

            idx += 1

    print('Finished Training')

def main():
    args = get_args_parser()    
    APPLICATION = "features_{}".format(args.epochs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    data_path = os.path.join(PATH, 'data')

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  

    model = SimpleCNN(DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(model, args.epochs, optimizer, criterion, train_loader, test_loader, APPLICATION)


if __name__ == "__main__":
    main()

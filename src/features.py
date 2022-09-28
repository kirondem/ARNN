
import argparse
import logging
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from models.associative_network import AssociativeNetwork
from models.feature_extractors import *
from models.model import Model

LOG_LEVEL = logging.getLevelName('DEBUG')
logging.basicConfig(level=LOG_LEVEL)

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args_parser():
    parser = argparse.ArgumentParser('Solar panel production prediction', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--lr', default=0.0001 , type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)

    parser.add_argument('--lr_a', default=0.001 , type=float)
    parser.add_argument('--time_steps', default=3, type=int)
    parser.add_argument('--no_of_units', default=49, type=int, help='Number of units in the associative network')
    parser.add_argument('--no_of_input_units', default=49, type=int, help='Number of input units')


    return parser.parse_args()

def train(model, model_vgg16, assoc_network, epochs, optimizer, criterion, time_steps, lr_a, train_loader, test_loader, APPLICATION):

    decay_threshold = 0.1
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        idx = 0

        learning_rate = lr_a * (1 - epoch / epochs)

        for batch in train_loader:
            optimizer.zero_grad()

            X, y = batch
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            #Extract features
            features = model_vgg16(X)

            # Associate features
            for i in range(features.size(1)):
                print(i)
                f = features[:, i, :, :].view(-1)
                W, H, H_H, avg_w_list = assoc_network.learn(f, time_steps, learning_rate, decay_threshold) 
            
            H_H = torch.tensor(H_H).to(DEVICE)

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

    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize( 
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
        )
    ])

    batch_size = args.batch_size
    data_path = os.path.join(PATH, 'data')

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            
    # create vgg16 backbone
    model_vgg16 = get_feature_extractor_backbone()
    model_vgg16 = model_vgg16.to(DEVICE)
    model_vgg16.eval()

    model = Model().to(DEVICE)
    optimizer_pars = {'lr': args.lr}
    model_params = list(model.parameters())
    optimizer = torch.optim.Adam(model_params, **optimizer_pars)
    criterion = nn.CrossEntropyLoss()

    assoc_network = AssociativeNetwork(args.no_of_units, args.time_steps)
    lr_a = args.lr_a

    train(model, model_vgg16, assoc_network, args.epochs, optimizer, criterion, args.time_steps, lr_a, train_loader, test_loader, APPLICATION)


if __name__ == "__main__":
    main()

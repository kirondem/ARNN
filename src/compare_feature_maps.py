
import torch
from lib import enums, constants, utils, plot_utils, cifar_features
from torchvision.models import resnet50, ResNet18_Weights
from torchvision import models
from numpy import dot
from numpy.linalg import norm

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def main():

    # Class names for the CIFAR-10 dataset
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    images, labels = cifar_features.getCIFAR10Data()

    images = images.to(device)
    # 10, 3, 32, 32
    # 6, 9, 9, 4, 1, 1, 2, 7, 8, 3
    # frog, truck, truck, deer, car, car, bird, horse, ship, cat

    #model = models.resnet18(weights='IMAGENET1K_V1')
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.layer3[1].register_forward_hook(get_activation('l3_conv2'))

    with torch.no_grad():
        output = model(images)
        features = activation['conv1'].cpu().numpy()
        features_l3_conv2 = activation['l3_conv2'].cpu().numpy()

    features_frog = features[0][1]
    features_truck = features[1][1]
    features_bird = features[6][1]
    features_cat = features[9][1]

    features_frog = features_frog.flatten()
    features_truck = features_truck.flatten()
    features_bird = features_bird.flatten()
    features_cat = features_cat.flatten()

    cos_sim = dot(features_truck, features_cat)/(norm(features_truck)*norm(features_cat))
    print(cos_sim)

if __name__ == '__main__':
    main()
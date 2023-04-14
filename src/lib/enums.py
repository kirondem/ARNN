from enum import Enum

class NetworkType(Enum):
    Sparse = 'Sparse'
    Dense = 'Dense'
    TabulaRasa = 'TabulaRasa'

class SynapticConnection(Enum):
    EE = 'EE'
    EI = 'EI'
    IE = 'IE'

class Phase(Enum):
    Plasticity = 'Plasticity'
    Training = 'Training'

class Application(Enum):
    base_line = 'base_line'


class FashionMNISTCategory(Enum):
    T_shirt_top = 0
    Trouser = 1
    Pullover = 2
    Dress = 3
    Coat = 4
    Sandal = 5
    Shirt = 6
    Sneaker = 7
    Bag = 8
    Ankle_boot = 9

class ANNNetworkType(Enum):
    DynamicLambda = 'dynamic_lambda'
    MagnitudeLamdaRelu = 'magnitude_lamda_relu'
    lambda_set_to_1 = 'lambda_set_to_1'



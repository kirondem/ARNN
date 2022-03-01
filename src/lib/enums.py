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



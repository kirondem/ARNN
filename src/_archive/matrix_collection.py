import numpy as np
from base import Base
from lib import enums
import random

class MatrixCollection(Base):

    def __init__(self, time_steps=1):
        super().__init__()

        self.W = [0] * time_steps   #Weights matrix
        self.H = [0] * time_steps   #Units matrix
        self.H_H = [0] * time_steps #Units associatively activated matrix
        self.DECAYED_IDXS = [0] * time_steps # decayed_activations_idxs
        self.DECAYED_ACTIVATIONS = [0] * time_steps # decayed_activations

        self.W[0] = np.zeros((super().no_of_units, super().no_of_units))
        self.H[0] = np.zeros(super().no_of_units)
        self.H_H[0] = []

        self.DECAYED_IDXS[0] = []
        self.DECAYED_ACTIVATIONS[0] = np.zeros(super().no_of_units)


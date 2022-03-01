import numpy as np

directly_activated_units_idxs = [180, 162, 177, 105, 121, 182, 124, 195, 23, 42, 79, 169, 12, 135]

decayed_activations_idxs =  [  2,   5,   8,  10,  12,  14,  15,  17,  20,  24,  25,  26,  32,
        43,  44,  48,  50,  54,  55,  58,  60,  63,  66,  69,  76,  78,
        81,  83,  86,  87,  90,  91,  94, 101, 103, 108, 110, 113, 121,
       131, 133, 134, 135, 139, 140, 146, 147, 150, 151, 158, 165, 174,
       176, 182, 193]

decayed_activations = [i for i in decayed_activations_idxs if i not in directly_activated_units_idxs]


print(decayed_activations)


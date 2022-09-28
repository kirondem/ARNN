from matplotlib import pyplot as plt
import numpy as np


def decay_activation_1(h):
    h = np.exp(-(h))
    return h

    
X =  np.arange(0, 5, 0.1)

y = decay_activation_1(X)

fig = plt.figure()
ax = fig.add_subplot(111)


xticks = [i for i in range(len(X))]

plt.ylabel("Activation")
plt.xlabel('Time'+ r'$\rightarrow$')
plt.plot(y)
plt.show()




import numpy as np
import matplotlib.pyplot as plt

def decay_activation(h, t, timesteps):
    h = h * (1 - t / timesteps)
    return h

def decay_activation_g(h, t, timesteps):
    h = h * np.exp(-(t**2 / timesteps))
    return h

time_steps = 10
h = 2
yticks = []
for i in range(time_steps):
    h = decay_activation(h, i, time_steps)
    yticks.append(h)


fig = plt.figure()
ax = fig.add_subplot(111)


xticks = [i for i in range(time_steps)]

plt.plot(xticks,yticks)
plt.show()



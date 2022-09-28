
import numpy as np
import matplotlib.pyplot as plt

def decay_activation(h, t, timesteps):
    h = h * ((1 - t) ** timesteps)
    return h

def decay_activation_g(h, t, timesteps):
    h = h * np.exp(-(t**2 / timesteps))
    return h

def decay_activation_1(constant, h, t, timesteps):
    h = np.exp(-(t))
    return h

time_steps = 30
h = 1
yticks = []
constant = 0.8
for i in range(time_steps):
    h = decay_activation(h, 0.01, time_steps)
    yticks.append(h)


fig = plt.figure()
ax = fig.add_subplot(111)


xticks = [i for i in range(time_steps)]

plt.ylabel("Activation")
plt.xlabel('Time'+ r'$\rightarrow$')
plt.plot(xticks,yticks)
plt.show()



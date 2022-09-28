# Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt

# Generating time data using arange function from numpy
time = np.arange(0, 10, 0.0001)
constant = 0.8

# Finding amplitude at each time
amplitude_grow = constant * np.exp(time)
amplitude_decay = constant * np.exp(-time)

# Plotting time vs amplitude using plot function from pyplot
plt.plot(time, amplitude_decay)

# Setting x axis label for the plot
plt.xlabel('Time'+ r'$\rightarrow$')

# Setting y axis label for the plot
plt.ylabel('Activation '+ r'$\rightarrow$')

# Showing legends
plt.legend(['Decaying Exponential'])

# Finally displaying the plot
plt.show()
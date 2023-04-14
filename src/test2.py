import numpy as np

t_s = 1.1794736253538114e-06
t_ds = 2.5255104577163505e-07

print(t_s > t_ds)


t_s = np.array([5.768522877831369e-20,
3.7573243204917986e-20,
6.140061595994472e-20,
6.46583793001913e-20,
5.1366878304157975e-20
])
t_ds = np.array([
    2.0817914705399199e-19,
    1.010814600709302e-19,
    2.0356812725617992e-19,
    1.024311303915262e-19,
    2.2400053083543997e-19
])

ts_avg = np.mean(t_s)
tds_avg = np.mean(t_ds)

print(tds_avg < ts_avg)

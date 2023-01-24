import perfplot
import numpy as np

def use_append(size):
    out = []
    for i in range(size):
        out.append(i)
    return out

def list_compr(size):
    return [i for i in range(size)]

def list_range(size):
    return list(range(size))

perfplot.show(
    setup=lambda n: n,
    kernels=[
        use_append,
        list_compr,
        list_range,
        np.arange,
        lambda n: list(np.arange(n))
    ],
    labels=["use_append", "list_compr", "list_range", "numpy", "list_numpy"],
    n_range=[2**k for k in range(15)],
    xlabel="len(a)",
    equality_check=None
)
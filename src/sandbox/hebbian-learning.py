import numpy as np

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(1000)

# Create and scale dataset
X, _ = make_blobs(n_samples=500, centers=2, cluster_std=5.0, random_state=1000)

scaler = StandardScaler(with_std=False)
Xs = scaler.fit_transform(X)


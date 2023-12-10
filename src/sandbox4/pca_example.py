from sklearn.decomposition import PCA
import numpy as np


# Generate a dataset with 2000 samples, each having 1024 features

data = np.random.rand(2000, 1024)

# Use PCA to reduce the dataset's dimensionality
pca = PCA(n_components=512)
reduced_data = pca.fit_transform(data)

# Output the shape of the reduced data
reduced_data.shape

print(reduced_data)
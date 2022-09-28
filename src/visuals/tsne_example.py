import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 2, 0], [1, 2, 1]], dtype=np.float64)
X_embedded = TSNE(n_components=2,  init='random').fit_transform(X)
print( X_embedded)

target_names = ['1', '2', '3', '4', '5']
y = np.array([1, 2, 3, 4, 1])

print(y == 5)
target_ids = range(len(target_names))

plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, target_names):
    x= X_embedded[y == i, 0]
    y1  = X_embedded[y == i, 1]
    plt.scatter(x, y1, c=c, label=label)
plt.legend()
plt.show()
x= 1
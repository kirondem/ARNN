import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

import numpy as np
import seaborn as sns

PATH  =  os.path.dirname(os.path.abspath(__file__))
epochs = 1
time_steps = 30
APPLICATION = 55000 # 2, 55000

path = os.path.join(PATH, '..', 'saved_models', '{}_{}_{}_w.npy'.format(epochs, time_steps, APPLICATION))
data = np.load(path)


#plt.imshow(data[29],     interpolation='nearest')
#plt.show()

data_subset = data[20]

tsne = TSNE(n_components=2, verbose=0, perplexity=10, n_iter=5000)
tsne_results = tsne.fit_transform(data_subset)



#df_subset['tsne-2d-one'] = tsne_results[:,0]
#df_subset['tsne-2d-two'] = tsne_results[:,1]

data = {'tsne-2d-one': tsne_results[:,0], 'tsne-2d-two': tsne_results[:,1]}
df_subset = pd.DataFrame(data)
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    #hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)

plt.show()
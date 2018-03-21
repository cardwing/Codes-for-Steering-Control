import numpy as np
import h5py
from time import time
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

color_list = ['#6ccca0', '#6cccc0', '#6cc5cc', '#6cbdcc', '#6cadcc', '#6c98cc', '#6c88cc', '#6c75cc', '#826ccc', '#9f6ccc']

def plot_embedding(X, Y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], '.',
                 color= color_list[Y[i]],
                 fontdict={'weight': 'bold', 'size': 15})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# color = plt.cm.Set1(Y[i] / 10.)
with h5py.File('/home/cardwing/Desktop/good_high_level_feature.h5', 'r') as hf:
    data = hf['name-of-dataset'][:]

data = np.reshape(data, (5614, 2700))
label = []
with open('/home/cardwing/Desktop/label_tsne.txt', 'r') as f:
    for line in f.readlines():
        label.append(int(line))
data_tmp = np.zeros((1500, 2700))
label_tmp = []
data_sort = []
num_label = 0
count = 0

for i in range(10):
    label_sort = []
    for j in range(5614):
        if label[j] == i:
            label_sort.append(j)
    shuffle(label_sort)
    for j in range(150):
        data_tmp[j + i * 150, :] = data[label_sort[j], :]
        label_tmp.append(i)
        
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(data_tmp)
print(X_tsne.shape)

plot_embedding(X_tsne, label_tmp, 
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

plt.show()

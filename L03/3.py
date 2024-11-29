# Ex 3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def import_database():
    data = pd.read_csv('../shared/data/iris.csv', header=None)
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    return features, labels


def apply_pca(data, n_comp):
    pca = PCA(n_components=n_comp)
    comp = pca.fit_transform(data)
    return comp


def visualize_pca(comps, labels):
    unique_classes = np.unique(labels)
    plt.figure(figsize=(10, 10))
    for cls in unique_classes:
        indices = labels == cls
        plt.scatter(comps[indices, 0], comps[indices, 1], label=cls)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    features, labels = import_database()
    components = apply_pca(features, n_comp=2)
    visualize_pca(components, labels)

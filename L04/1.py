# Lab 4 - Kernel Methods. Classification

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix


def train_test_data():
    data = pd.read_csv('../shared/data/iris.csv', header=None)

    indices = data.index.tolist()
    random.shuffle(indices)

    train_indices = random.sample(indices, 100)
    test_indices = np.setdiff1d(indices, train_indices)

    train = data.loc[train_indices]
    test = data.loc[test_indices]

    train_d = train.iloc[:, :-1]
    train_l = train.iloc[:, -1]

    test_d = test.iloc[:, :-1]
    test_l = test.iloc[:, -1]

    return train_d, train_l, test_d, test_l


def train_model(kernel, train_d, train_l, test_d, test_l):
    clf = svm.SVC(kernel=kernel)
    clf.fit(train_d, train_l)
    estimated_labels = clf.predict(test_d)
    # print(estimated_labels)

    accuracy = accuracy_score(test_l, estimated_labels)
    classification_error = 1 - accuracy

    conf_matrix = confusion_matrix(test_l, estimated_labels)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="OrRd", xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'], cbar=False)
    plt.xlabel('Actual Class')
    plt.ylabel('Predicted Class')
    plt.title(f'kernel={kernel}')
    plt.show()

    print(f"kernel={kernel}, classification_error={classification_error}, confusion_matrix: \n {conf_matrix}")


if __name__ == '__main__':
    print("---------------------\na)")

    train_test_data()
    train_data, train_labels, test_data, test_labels = train_test_data()
    print(len(train_data))
    print(len(test_data))

    print("---------------------\nb) + c)")
    train_model("linear", train_data, train_labels, test_data, test_labels)

    print("---------------------\nd)")
    train_model("rbf", train_data, train_labels, test_data, test_labels)

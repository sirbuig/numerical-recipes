# Ex 3

import numpy as np
import matplotlib.pyplot as plt

def read_file():
    z = []
    w = []
    with open("regresie.csv", "r") as f:
        for line in f:
            line = line.strip().split(',')
            z.append(float(line[0]))
            w.append(float(line[1]))
    return z, w


def create_system(z, w):
    z = np.array(z)
    w = np.array(w)
    col = np.ones(len(z))
    a = np.vstack((z, col)).T
    return a, w

def plot(z, w, x):
    plt.scatter(z, w, color='green', marker='x')

    # alpha * x + beta
    line_x_range = np.linspace(min(z), max(z), 100)

    line_y_range = x[1] + x[0] * line_x_range

    plt.plot(line_x_range, line_y_range, color='purple')

    plt.show()

if __name__ == '__main__':
    z, w = read_file()

    print("---------------------\na)")
    A, b = create_system(z, w)

    print(A)
    print(b)
    # print(A.shape)
    # print(b.shape)

    print("---------------------\nb)")
    x, _, _, _ = np.linalg.lstsq(A, b)
    print(x)

    print("---------------------\nc)")
    plot(z, w, x)


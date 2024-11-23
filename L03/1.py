# Ex 1

import numpy as np
import numpy.linalg as la
import random

def matrix_rank(a):
    return la.matrix_rank(a)

def add_columns(a, n_, c_):
    nr = n_ - a.shape[1]
    coeff = random.sample(range(100), nr)
    columns = []
    for i in range(nr):
        column = np.zeros(a.shape[0])
        for j in range(c_):
            column += coeff[i] * a[:, j]
        columns.append(column)
    columns = np.column_stack(columns)
    a = np.hstack((a, columns))

    return a

def add_noise(a, mean, d):
    noise = np.random.normal(mean, d, a.shape)
    return a + noise

if __name__ == '__main__':
    m = 10
    r = 4
    matrix = np.random.randn(m,r)

    print("---------------------\na)")
    print(matrix_rank(matrix))

    print("---------------------\nb)")
    n = 8
    c = 3
    matrix_2 = add_columns(matrix, n, c)
    print(matrix_rank(matrix_2))

    print("---------------------\nc)")
    noisy_matrix = add_noise(matrix_2, 0, 0.2)
    print(matrix_rank(noisy_matrix))

    print("---------------------\nd)")
    u, s, v = la.svd(noisy_matrix)
    print(s)
# Ex 1

import numpy as np
import numpy.linalg as la

# Eigenvalue power method
def power_method(a, tol, max_iter):
    # a = np.array(a)
    y = np.random.randn(a.shape[0])
    iteration = 0
    err = 1

    while err > tol:
        if iteration > max_iter:
            break

        y = y / la.norm(y)
        z = a.dot(y)
        z = z / la.norm(z)
        err = abs(1 - abs(z.transpose().dot(y)))
        y = z
        iteration += 1

    eigenvalue = np.dot(y.T, np.dot(a,y)) / np.dot(y.T, y)
    return y, eigenvalue

if __name__ == '__main__':
    n = 6
    matrix = np.random.randn(n, n)
    # matrix = [[1, 2, 3],
    #           [4, 5, 6],
    #           [7, 8, 9]]

    solution, eigenvalues = la.eig(matrix)
    solution_2, value = power_method(matrix, tol=0, max_iter=1000)

    print(solution)
    print("\n")
    print(eigenvalues)
    print(solution_2)
    print(value)
# Ex 2

import numpy as np

# Gaussian elimination

"""
Function that transform A, b from a given Ax=b to an upper triangular matrix.
"""


def transform(a, rhs, dim):
    for k in range(0, dim - 1):
        for i in range(k + 1, dim):
            a[i][k] = -a[i][k] / a[k][k]
            rhs[i] = rhs[i] + rhs[k] * a[i][k]
            for j in range(k + 1, dim):
                a[i][j] = a[i][j] + a[k][j] * a[i][k]
    return np.triu(a), rhs


"""
Function that solves upper triangular system with back substitution
"""


def utris(l, rhs):
    sol = rhs.copy()
    for i in range(len(l)-1, -1, -1):
        for j in range(i + 1, len(l)):
            sol[i] = sol[i] - l[i][j] * sol[j]
        sol[i] = sol[i] / l[i][i]
    return sol


if __name__ == '__main__':
    # n = 3
    # A = [[2, 4, -2],
    #      [4, 9, -3],
    #      [-2, -3, 7]]
    # b = np.array([2, 8, 10])

    A = np.random.randn(6, 6)
    b = np.random.randn(6)

    print("Original: ")
    print("A=", A)
    print("b=", b)

    # row echelon form
    A_gauss, b_gauss = transform(A, b, 6)
    print("---------------------\na)")
    print("In row echelon form: ")
    print("A=", A_gauss)
    print("b=", b_gauss)

    print("---------------------\nb), c)")
    x = utris(A_gauss, b_gauss)
    print("x=", x)

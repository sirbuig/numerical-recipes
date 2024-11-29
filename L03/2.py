# Ex 2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def img_svd(path):
    # load image
    image = mpimg.imread(path)
    # calculate SVD
    u, s, v = np.linalg.svd(image)
    return image, u, s, v


def k_aprox(u_orig, s_orig, v_orig, k):
    U_k = u_orig[:, :k]
    S_k = np.diag(s_orig[:k])
    V_k = v_orig[:k, :]
    k_image = np.dot(U_k, np.dot(S_k, V_k))

    return k_image


if __name__ == '__main__':
    print("---------------------\na)")
    original_img, U, S, V = img_svd('../shared/media/Baboon.bmp')
    plt.title('Original Image')
    plt.imshow(original_img, cmap='gray')
    plt.show()
    # print("U: ", U)
    # print("S: ", S)
    # print("V: ", V)

    print("---------------------\nb) + c)")
    m, n = U.shape
    print("M = %d x %d" % (m, n))
    approx_image = k_aprox(U, S, V, 50)
    plt.title('Approximation Image with k=50')
    plt.imshow(approx_image, cmap='gray')
    plt.show()

    print("---------------------\nd)")
    k_list = [10, 20, 50, 100, 500]
    for k in k_list:
        approx_image = k_aprox(U, S, V, k)
        plt.title('Approximation Image with k=%d' % k)
        plt.imshow(approx_image, cmap='gray')
        plt.show()

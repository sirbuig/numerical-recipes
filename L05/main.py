# Lab 5 - Dictionary Learning

import numpy as np
from dictlearn import DictionaryLearning
from dictlearn import methods
from matplotlib import image, pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.preprocessing import normalize

p = 8  # patch dimension (nr of pixels)
s = 10  # sparsity
N = 1849  # total nr of patches
n = 256  # nr of atoms from the dictionary
K = 50  # nr of DL iterations
sigma = 0.075  # standard deviation of the noise


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 0
    max_pixel = 255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


if __name__ == '__main__':
    print("---------------------\n1)")
    # a)
    I = image.imread('../shared/media/Lenna.png')
    I = rgb2gray(I)
    I = I[250:300, 250:300]
    print(f"Shape of the image: {I.shape}")
    plt.imshow(I, cmap='gray')
    plt.show()

    # b)
    Inoisy = I + sigma * np.random.randn(I.shape[0], I.shape[1])
    plt.imshow(Inoisy, cmap='gray')
    plt.show()

    # c)
    Ynoisy = extract_patches_2d(Inoisy, (p, p))
    print(f"After extracting the patches: {Ynoisy.shape}")
    Ynoisy = Ynoisy.reshape(Ynoisy.shape[0], -1)
    print(f"After reshaping: {Ynoisy.shape}")

    Ynoisy = Ynoisy.transpose()
    print(f"After transpose: {Ynoisy.shape}")

    Ynoisy_signals_mean = np.mean(Ynoisy, axis=0, keepdims=True)
    print(f"Mean shape: {Ynoisy_signals_mean.shape}")
    Ynoisy -= Ynoisy_signals_mean

    # d)
    print(Ynoisy.shape)
    random_idx = np.random.choice(Ynoisy.shape[1], N)
    Y = Ynoisy[:, random_idx]
    print(Y.shape)

    print("---------------------\n2)")
    # a)
    D0 = np.random.randn(p * p, n)
    print(D0.shape)  # (64, 256)
    D0 = normalize(D0, axis=0, norm='max')

    # b)
    dl = DictionaryLearning(
        n_components=n,
        max_iter=K,
        fit_algorithm='ksvd',
        n_nonzero_coefs=s,
        code_init=None,
        dict_init=D0,
        params=None,
        data_sklearn_compat=False
    )
    dl.fit(Ynoisy)
    D = dl.D_
    print(D.shape)  # (64, 256)

    print("---------------------\n3)")
    # a)
    Xc, err = methods.omp(Ynoisy, D, n_nonzero_coefs=s)
    print(Xc.shape)  # (256, 1000)

    # b)
    # np.dot(D, Xc) => shape (64, 1000)
    # np.dot(D, Xc).T => shape (1000, 64)
    Yc = np.dot(D, Xc).T + Ynoisy_signals_mean.T
    print(Yc.shape)  # (1000, 64)

    # c)
    Yc = np.reshape(Yc, shape=(-1, p, p))
    Ic = reconstruct_from_patches_2d(Yc, I.shape)
    plt.imshow(Ic, cmap='gray')
    plt.show()

    # print("---------------------\n4)")
    # a), b)
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(I, cmap='gray')
    axarr[0].set_title('Original')

    axarr[1].imshow(Inoisy, cmap='gray')
    axarr[1].set_title(f'Noisy\n psnr={psnr(I, Inoisy):.2f}')

    axarr[2].imshow(Ic, cmap='gray')
    axarr[2].set_title(f'Reconstructed\n psnr={psnr(I, Ic):.2f}')

    plt.tight_layout()
    plt.show()

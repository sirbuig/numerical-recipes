# Ex 2
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

def load_image(path):
    img = np.asarray(Image.open(path))
    imgplot = plt.imshow(img)
    plt.colorbar()

if __name__ == '__main__':
    load_image('Baboon.bmp')
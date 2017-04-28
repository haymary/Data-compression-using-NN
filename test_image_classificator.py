from ImageClassificator import ImageClassificator
from scipy import misc
import numpy as np

# img = np.random.rand(5,5)

img = misc.imread("pics/blocks/rand_ok.jpg", flatten=True, mode = "L")
img = img / 255.

from BlockAutoencoder import BlockAutoencoder

x_simple = np.array([img])
simple = BlockAutoencoder()
simple.train(x_simple, 5, 1, 5, 100, 200, 0.3)

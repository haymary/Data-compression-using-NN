from ImageClassificator import ImageClassificator
from scipy import misc
import numpy as np

# img = np.random.rand(5,5)

img = misc.imread("pics/blocks/rand_ok.jpg", flatten=True, mode = "L")
img = img / 255.
img = np.zeros([5,5])
classificator = ImageClassificator()

img_class = classificator.getImageClass(img)

print(img_class)

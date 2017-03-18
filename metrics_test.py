from metrics import print_metrics
from PIL import Image
import numpy as np

image = Image.open('/home/maria/Documents/test/pics/lena.jpg')
image = image.convert('L') # convert the image to *greyscale*
image = np.array(image)
print_metrics(image, image, image)

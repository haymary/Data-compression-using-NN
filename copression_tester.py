# from __future__ import print_function
from SimpleAutoencoder import SimpleAutoencoder
import pylab as plt
import numpy as np
from PIL import Image
from metrics import print_metrics
from scipy import ndimage, misc


# face = misc.face()
# img = misc.imread("pics/cat.jpg", flatten=True, mode = "L")
# img2 = misc.imread("pics/lena.jpg", flatten=True, mode = "L")
# image_list = (img,img2)

import glob
image_list = []
for filename in glob.glob('pics/sample_sm/*.jpg'):
	print "Loading {}".format(filename)
	im = misc.imread(filename, flatten=True, mode = "L")
	image_list.append(im)
image_list =  tuple(image_list)

auto = SimpleAutoencoder()



auto.train(image_list, 15, 1, 5, 15, 256, 0.3)
auto.save_models()

auto.load_models()
r = auto.encode("pics/lena.jpg")
s = auto.decode(r)

# plt.imshow(s, cmap='gray')
# plt.show()

result = Image.fromarray((s * 255).astype(np.uint8))
result.save("pics/s.jpg")

image = misc.imread("pics/lena.jpg", flatten=True, mode = "L")
# image = Image.open('/home/maria/Documents/test/pics/lena.jpg')
# image = image.convert('L')
# image = np.array(image)
r = np.array(r)
s = np.array(s)
print_metrics(image, r, s)

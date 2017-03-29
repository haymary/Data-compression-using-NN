from skimage import measure, filters
from scipy import ndimage, misc
import pylab as plt
import numpy as np

n = 20
l = 256
# im = np.zeros((l, l))
# points = l * np.random.random((2, n ** 2))
# im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
# im = filters.gaussian_filter(im, sigma=l / (4. * n))
# blobs = im > im.mean()
# all_labels = measure.label(blobs)
# plt.imshow(all_labels, cmap='gray')
# plt.contour(all_labels, [0.5]) 
# plt.show()

im = misc.imread("pics/lena.jpg", flatten=True, mode = "L")
# plt.imshow(image, cmap='gray')
# plt.contour(image, [0.5]) 
# plt.show()
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
image = im
image = filters.gaussian(image, sigma=1)
image = image > image.mean()
all_labels = measure.label(image, connectivity = 2)
plt.imshow(all_labels, cmap='gray')
plt.contour(all_labels, [0.5]) 
plt.show()
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
# from skimage import io
import math
# from skimage.data import astronaut
# from skimage.color import rgb2gray
# from skimage.filters import sobel
from skimage.segmentation import felzenszwalb
# from skimage.util import img_as_float

def blocks_to_image(blocks, image_shape, box_size):
    x, y = 0, 0
    image = []
    row = []
    for i in range(0, len(blocks)):
        box = to_square(blocks[i], box_size)
        if(len(row) == 0):
            row = box
            row = np.array(row)
        else:
            row = add_box(row, box)
        if(row.shape[1] >= image_shape[1]):
            if(len(image) == 0):
                image = row[0 : image_shape[0], 0 : image_shape[1]]
            else:
                image = np.vstack((image, row[0 : image_shape[0] - image.shape[0], 0 : image_shape[1]]))
            row = []
    return image
def to_square(matrix, box_size):
    return np.reshape(matrix, (box_size, box_size))

def add_box(row, box):
    new_row = list()
    for i in range(0, row.shape[0]):
        new_row.append(np.append(row[i], box[i]))
    new_row = np.array(new_row)
    return new_row

def image_to_blocks(img, box_size, image_shape):
    x_n_boxes = int(math.ceil(image_shape[1] / float(box_size) ))
    y_n_boxes = int(math.ceil(image_shape[0] / float(box_size) ))

    result = np.zeros((x_n_boxes*y_n_boxes, box_size * box_size))
    res = list()
    for i in xrange(image_shape[0]):
        for j in xrange(image_shape[1]):
            x = j % box_size
            y = i % box_size
            el_num = int(x + box_size * y)
            block_num = int(math.floor(j/box_size) + (x_n_boxes * math.floor(i/box_size)))
            result[block_num,el_num] = img[i][j]

    res = list()
    for r in result:
        b = list()
        b.append(r)
        res.append(blocks_to_image(b, tuple([5, 5]), 5))
    return res

# img = img_as_float(io.imread("pics/lena.jpg"))
# print (img)

from scipy import ndimage, misc

# img = misc.imread("pics/lena.jpg", flatten=True, mode = "L")
# img = img / 255.
# blocks = image_to_blocks(img,5,img.shape)
# # print (blocks)
# img = blocks[500]

# print (img)
# misc.imsave('pics/blocks/3.jpg', img)

# img = np.random.rand(5,5)
# misc.imsave('pics/blocks/rand_ok.jpg', img)

img = misc.imread("pics/blocks/rand.jpg", flatten=True, mode = "L")
img = img / 255.

# img = np.zeros([5,5])

segments_fz = felzenszwalb(img, scale=5, sigma=0.5, min_size=1)

print (segments_fz)

print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print("Felzenszwalb number of segments: {}".format(len((segments_fz))))

plt.imshow(img)
plt.show()

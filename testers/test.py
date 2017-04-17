from PIL import Image
import numpy as np
from working_with_images import *
from alg import *

def split_into_squares(img, box_size):
    # Define the window size
    windowsize_r, windowsize_c = box_size, box_size

    image = list()
    # Crop out the window
    for r in range(0, img.shape[0] + (windowsize_r - 1), windowsize_r):
        for c in range(0, img.shape[1] + (windowsize_c - 1), windowsize_c):
            x = r + windowsize_r
            y = c + windowsize_c
            if(x >= img.shape[0]):
                x = img.shape[0]
            if(y >= img.shape[1]):
                y = img.shape[1]
            window = img[r : x, c : y]
            window = np.hstack((window, np.zeros((window.shape[0], windowsize_c - window.shape[1]))))
            window = np.vstack((window, np.zeros((windowsize_r - window.shape[0], windowsize_c))))
            window = window.ravel()
            original_shape = len(window)
            window = np.reshape(window, (1, original_shape))
            image.append(window)
    return image
def to_square(martix, box_size):
    return np.reshape(martix, (box_size, box_size))

def add_box(row, box):
    new_row = list()
    for i in range(0, row.shape[0]):
        new_row.append(np.append(row[i], box[i]))
    new_row = np.array(new_row)
    return new_row

def back_to_image(boxes, image_shape, box_size):
    x, y = 0, 0
    image = []
    row = []
    for i in range(0, len(boxes)):
        box = to_square(boxes[i], box_size)
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
import math
def image_to_blocks(img, box_size, image_shape):
    x_n_boxes = int(math.ceil(image_shape[1] / float(box_size) ))
    y_n_boxes = int(math.ceil(image_shape[0] / float(box_size) ))

    result = np.zeros((x_n_boxes*y_n_boxes, box_size * box_size))

    for i in xrange(image_shape[0]):
        for j in xrange(image_shape[1]):
            x = j % box_size
            y = i % box_size
            el_num = int(x + box_size * y)
            block_num = int(math.floor(j/box_size) + (x_n_boxes * math.floor(i/box_size)))
            result[block_num,el_num] = img[i][j]
    return result

import pylab as plt
# image = np.random.randint(0, 256, size=(10, 10))
image = Image.open('/home/maria/Documents/test/pics/cat.jpg')
image = image.convert('L') # convert the image to *greyscale*
image = np.array(image)

blocks = image_to_blocks(image, 5, image.shape)
back = back_to_image(blocks, image.shape, 5)

plt.imshow(image)
plt.show()

plt.imshow(back)
plt.show()

result = Image.fromarray((back * 255).astype(np.uint8))
result.save("/home/maria/Documents/test/pics/i2.jpg")

# image = Image.open('/home/maria/Documents/test/pics/lena.jpg')
# # image = image.convert('L') # convert the image to *greyscale*
# im_array = np.array(image)
# image_shape = im_array.shape
# x_axis = image_shape[0]
# y_axis = image_shape[1]
# # x = im_array.ravel()
# # original_shape = len(x)
# print "Original Img"
# print im_array
# box_size = 30
# x = split_into_squares(im_array, box_size)
#
#
# image = back_to_image(x, image_shape, box_size)
# print image
#
# show_immgs(im_array, image)

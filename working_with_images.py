from PIL import Image
import numpy as np
import math

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
    return result

from PIL import Image
import numpy as np

def split_into_squares(img, box_size):
    # Define the window size
    windowsize_r, windowsize_c = box_size, box_size

    image = list()
    # Crop out the window
    r_end = img.shape[0] + (windowsize_r - 2)
    c_end = img.shape[1] + (windowsize_c - 2)
    if (img.shape[0] % box_size == 0):
        r_end = img.shape[0] - 1
    if (img.shape[1] % box_size == 0):
        c_end = img.shape[1] - 1

    for r in range(0, r_end, windowsize_r):
        for c in range(0, c_end, windowsize_c):
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

def show_immgs(orig_img, decoded_img):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 4))
    # display original
    ax = plt.subplot(2, 2, 1)
    plt.imshow(orig_img)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, 2, 2)
    plt.imshow(decoded_img)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()

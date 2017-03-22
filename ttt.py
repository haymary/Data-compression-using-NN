import numpy as np
from PIL import Image
from metrics import print_metrics, write_to_file
from scipy import ndimage, misc

image = misc.imread("pics/lena.jpg", flatten=True, mode = "L")
block_size = 15
layer_step = 1
number_of_layers = 5
epoch = 15
batch_size = 200 
split_rate = 0.3

print_metrics(image, image, image)
write_to_file("", image, image, image, 20, block_size, layer_step, number_of_layers, epoch, batch_size, split_rate)

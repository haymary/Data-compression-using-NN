# from __future__ import print_function
from SimpleAutoencoder import SimpleAutoencoder
import pylab as plt
import numpy as np
from PIL import Image
from metrics import print_metrics, write_to_file
from scipy import ndimage, misc
import os

#params
block_size = 15
layer_step = 1
number_of_layers = 5
number_of_epoches = 15
batch_size = 200
split_rate = 0.3

blocks = [x for x in range(15, 50, 10)]
num_epoch = [x for x in range(100, 200, 20)]
num_layers = [x in range(2, 10, 1)]
num_steps = [x in range(1, 9, 1)]

# Load data
import glob
image_list = []
for filename in glob.glob('pics/sample_sm/*.jpg'):
	print "Loading {}".format(filename)
	im = misc.imread(filename, flatten=True, mode = "L")
	image_list.append(im)
image_list =  tuple(image_list)

num_try = 12
for block_size in blocks:
    for epoches in num_epoch:
        for number_of_layers in num_layers:
            for layer_step in num_steps:
            	path_to_result = "results/" + str(num_try) + "/"
            	os.makedirs(path_to_result)

            	auto = SimpleAutoencoder()
            	auto.train(image_list, block_size, layer_step, number_of_layers, epoches, batch_size, split_rate)
            	auto.save_models(num = str(num_try))
            	# auto.load_models(num = str(num_try))

            	image = misc.imread("pics/lena.jpg", flatten=True, mode = "L")
            	r = auto.encode(image)
            	s = auto.decode(r)

            	# plt.imshow(s, cmap='gray')
            	# plt.show()

            	result = Image.fromarray((s * 255).astype(np.uint8))
            	result.save(path_to_result + "decoded.jpg")
                misc.imsave(path_to_result + 'using_imsave.jpg', s)

            	r = np.array(r)
            	s = np.array(s)

            	print_metrics(image, r, s)
            	write_to_file(path_to_result, image, r, s, len(image_list), block_size, layer_step, number_of_layers, epoches, batch_size, split_rate)

                num_try += 1

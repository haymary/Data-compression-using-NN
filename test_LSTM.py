from LSTMAutoencoder import LSTMAutoencoder
import pylab as plt
import numpy as np
from PIL import Image
from metrics import print_metrics, write_to_file
from scipy import ndimage, misc
import os

path_to_result = "results/LSTM" + "" + "/"
if not os.path.exists(path_to_result):
    os.makedirs(path_to_result)

#params
block_size = 8
layer_step = 1
number_of_layers = 5
number_of_epoches = 15
batch_size = 200
split_rate = 0.3
epoches = 15

#train data
image = misc.imread("pics/cat.jpg", flatten=True, mode = "L")
image_list = [image]
image_list =  tuple(image_list)

#train
auto = LSTMAutoencoder()
auto.train(image_list, block_size, layer_step, number_of_layers, epoches, batch_size, split_rate)
auto.save_models(num = str(num_try))
# auto.load_models(num = str(num_try))

#test
image = misc.imread("pics/lena.jpg", flatten=True, mode = "L")
r = auto.encode(image)
s = auto.decode(r)

result = Image.fromarray((s * 255).astype(np.uint8))
result.save(path_to_result + "decoded.jpg")
misc.imsave(path_to_result + 'using_imsave.jpg', s)

r = np.array(r)
s = np.array(s)

print_metrics(image, r, s)
write_to_file(path_to_result, image, r, s, len(image_list), block_size, layer_step, number_of_layers, epoches, batch_size, split_rate)

num_try += 1

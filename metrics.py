from sklearn.metrics import mean_squared_error
import math
import numpy as np
from PIL import Image

def mse(A, B):
    return ((A - B) ** 2).mean(axis=None)
    #mean_squared_error(orig_img, comp_img)

# From Optimized Image Compression through Artificial Neural Networks and Wavelet Theory
def psnr(m, A, B):
    tmp = np.sum(A)/len(A)
    nmse = m/(np.sum(A)/len(A) * np.sum(B)/len(B))
    return 10 * math.log10(1/nmse)

# compression ratio = orig file size / compressed file size
def compression_ratio(orig_img_size, comp_img_size):
    return orig_img_size / comp_img_size

def get_metrics(A, B, C):
    m = mse(A, C)
    p = psnr(m, A, C)
    cr = compression_ratio(A.shape[0] * A.shape[1], B.shape[0] * B.shape[1])
    return m, p, cr

def print_metrics(orig_img, comp_img, decoded_img):
    mse, psnr, comp_ratio = get_metrics(orig_img, comp_img, decoded_img)
    print "MSE = ", mse
    print "PSNR = ", psnr
    print "Compression ratio: ", comp_ratio

def write_to_file(path_to_results, orig_img, comp_img, decoded_img, train_set_size, block_size, layer_step, number_of_layers, epoch, batch_size, split_rate):
    mse, psnr, comp_ratio = get_metrics(orig_img, comp_img, decoded_img)
    file = open(path_to_results + 'result.txt', 'w+')
    file.write("------PARAMS------" + "\n")
    file.write("Size of trainse = {} \n".format(train_set_size) )
    file.write("block_size = " + str(block_size) + "\n")
    file.write("layer_step = " + str(layer_step) + "\n")
    file.write("number_of_layers = " + str(number_of_layers) + "\n")
    file.write("epoch = " + str(epoch) + "\n")
    file.write("batch_size = " + str(batch_size) + "\n")
    file.write("split_rate = " + str(split_rate) + "\n")
    file.write("------RESULTS-----" + "\n")
    file.write("MSE = " + str(mse) + "\n")
    file.write("PSNR = " + str(psnr) + "\n")
    file.write("Compression ratio: " + str(comp_ratio) + "\n")
    file.close()
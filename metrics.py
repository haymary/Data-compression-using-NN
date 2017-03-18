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

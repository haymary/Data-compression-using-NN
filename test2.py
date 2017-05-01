from SimpleAutoencoder import SimpleAutoencoder
import numpy as np
from PIL import Image
from scipy import misc

num_try = 9
path_to_result = "results/test/"
e_name = path_to_result + 'enc' + str(num_try) + '.jpg'

auto = SimpleAutoencoder()
auto.load_models(num = str(num_try))

image = misc.imread("pics/lena.jpg", flatten=True, mode = "L")
r = auto.encode(image)


a = np.reshape(r, (102400, 25))
r_new = np.array(a)


misc.imsave(e_name, r_new)
image = misc.imread(e_name, flatten=True, mode = "L")
r = np.reshape(image, 102400,1,25)
s = auto.decode(r)


s = np.array(s)
misc.imsave(path_to_result + 'dec' + str(num_try) + '.jpg', s)

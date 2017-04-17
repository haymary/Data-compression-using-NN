from working_with_images import *
from keras.datasets import mnist
import pylab as plt
import numpy as np

# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.0
# im_array = np.array(x_train)

im_size = 9
box_size = 3
# image = np.random.randint(0, 256, size=(im_size,im_size))

image = Image.open('/home/maria/Documents/test/pics/cat.jpg')
image = image.convert('L') # convert the image to *greyscale*
im_array = np.array(image)
image_shape = im_array.shape

print im_array
x = split_into_squares(im_array, box_size)
print x

i2 = back_to_image(x, image_shape, box_size)
print i2

plt.imshow(i2)
plt.show()

result = Image.fromarray((i2 * 255).astype(np.uint8))
result.save("/home/maria/Documents/test/pics/i2.jpg")

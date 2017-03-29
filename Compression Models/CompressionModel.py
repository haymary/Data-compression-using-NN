import abc
import math

from scipy import ndimage, misc

from keras.layers import Input, Dense
from keras.models import Model

from keras.models import load_model

import numpy as np

from keras.datasets import mnist

class CompressionModel:

    #BEGIN VARS
    encoder = ""
    decoder = ""
    encoding_dim = 0
    image_shape = ()
    input_image_size = 0
    block_size = 0
    #END VARS

    #BEGIN ABSTRACT METHODS
    @abc.abstractmethod
    def train(self, train_set, block_size, layer_step, number_of_layers, epoch, batch_size, split_rate):
        pass

    def encode(self, img):

        # img = misc.imread(image, flatten=True)

        self.image_shape = img.shape
        blocks = self.image_to_blocks(img, self.block_size, img.shape)
        for i in range(0, len(blocks)):
            blocks[i] = blocks[i].astype('float32') / 255.0
            #blocks[i] = blocks[i].reshape((len(blocks[i]), np.prod(blocks[i].shape[1:])))

        predicted = list()
        for i in blocks:
            g = np.reshape(i, (1, len(i)))
            predicted.append(self.encoder.predict(g))
        return predicted

    def decode(self, encoded_imgs):
        blocks = list()
        for i in encoded_imgs:
            blocks.append(self.decoder.predict(i))
        image = self.blocks_to_image(blocks, self.image_shape, self.block_size)
        return image

    def save_models(self, num = ""):
        self.decoder.save("models/decoder" + num + ".h5")
        self.encoder.save("models/encoder" + num + ".h5")
        f = open('models/headers' + num + '.txt', 'w')
        f.write(str(self.block_size) + "\n")
        f.close()

    def load_models(self, num = ""):
        self.decoder = load_model("models/decoder" + num + ".h5")
        self.encoder = load_model("models/encoder" + num + ".h5")
        f = open('models/headers' + num + '.txt', 'r')
        for line in f:
            data = line.split()
            self.block_size = int(data[0])

    #END ABSTRACT METHODS

    #BEGIN GENERAL METHODS
    def init_values(self, train_set, block_size, layer_step, number_of_layers):
        self.encoding_dim = block_size * block_size - ((number_of_layers) * layer_step)
        self.input_image_size = block_size * block_size
        self.block_size = block_size

    def image_to_blocks(self, img, box_size, image_shape):
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

    def blocks_to_image(self, blocks, image_shape, box_size):
        x, y = 0, 0
        image = []
        row = []
        for i in range(0, len(blocks)):
            box = self.to_square(blocks[i], box_size)
            if(len(row) == 0):
                row = box
                row = np.array(row)
            else:
                row = self.add_box(row, box)
            if(row.shape[1] >= image_shape[1]):
                if(len(image) == 0):
                    image = row[0 : image_shape[0], 0 : image_shape[1]]
                else:
                    image = np.vstack((image, row[0 : image_shape[0] - image.shape[0], 0 : image_shape[1]]))
                row = []
        return image

    def get_sample_dataset(self):
        (x_train, _), (x_test, _) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        return x_train, x_test


    def to_square(self, matrix, box_size):
        return np.reshape(matrix, (box_size, box_size))

    def add_box(self, row, box):
        new_row = list()
        for i in range(0, row.shape[0]):
            new_row.append(np.append(row[i], box[i]))
        new_row = np.array(new_row)
        return new_row


    def split_list(self, my_list, percent):
        s = len(my_list)
        A = my_list[0:int(s*percent)]
        B = my_list[int(s*percent):s]
        return A, B

    #END GENERAL METHODS

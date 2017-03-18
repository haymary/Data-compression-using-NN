from CompressionModel import CompressionModel

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

from keras.datasets import mnist

class SimpleAutoencoder(CompressionModel):
    def train(self, train_set, block_size, layer_step, number_of_layers, epoch, batch_size, split_rate):

        self.init_values(train_set, block_size, layer_step, number_of_layers)

        input_img = Input(shape=(self.input_image_size,))
        num_l = self.input_image_size - layer_step
        encoded = Dense(num_l, activation='relu')(input_img) #23

        for i in range(1, number_of_layers): #21 19 17
            num_l -= layer_step
            encoded = Dense(num_l, activation='relu')(encoded)

        num_l += layer_step
        decoded = Dense(num_l, activation='relu')(encoded) #19

        for i in range(0, number_of_layers-1): #21 23 25
            num_l += layer_step
            decoded = Dense(num_l, activation='relu')(decoded)


        autoencoder = Model(input=input_img, output=decoded)
        # print autoencoder.summary()
        # self.encoder = Model(input=input_img, output=encoded)
        #
        # encoded_input = Input(shape=(self.encoding_dim,))
        # self.decoder = Model(input=encoded_input, output=autoencoder.layers[-1](encoded_input))
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.encoder = autoencoder
        self.decoder = autoencoder

        x_train = list()
        for i in train_set:
            # print "Processing {}".format(i)
            img = self.image_to_blocks(i, block_size, i.shape)
            # x_train = x_train + img
            if len(x_train) == 0:
                x_train = img
            else:
                x_train = np.vstack((x_train, img))

        x_test, x_train = self.split_list(x_train, split_rate)
        x_test = np.array(x_test)
        x_train = np.array(x_train)

        for i in range(0, len(x_train)):
            x_train[i] = np.array(x_train[i]).astype('float32') / 255.0
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

        for i in range(0, len(x_test)):
            x_test[i] = np.array(x_test[i]).astype('float32') / 255.0
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        #x_train, x_test = self.get_sample_dataset()


        autoencoder.fit(x_train, x_train,
                        nb_epoch=epoch,
                        batch_size=batch_size,
                        shuffle=True,   
                        validation_data=(x_test, x_test))

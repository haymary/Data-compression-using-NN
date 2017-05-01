from CompressionModel import CompressionModel

from keras.layers import Input, Dense, LSTM, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np

from keras.datasets import mnist

class LSTMAutoencoder(CompressionModel):
    def train(self, train_set, block_size, layer_step, number_of_layers, epoch, batch_size, split_rate):

        self.init_values(train_set, block_size, layer_step, number_of_layers)
        input_img = Input(shape=(block_size, block_size,))
        num_l = self.input_image_size - layer_step

        encoded = LSTM(num_l,  activation='relu')(input_img)
        decoded = LSTM(self.input_image_size,  activation='relu')(encoded)
        # for i in range(1, number_of_layers):
        #     num_l -= layer_step
        #     #x = MaxPooling2D((2, 2), border_mode='same')(x)
        #     x = LSTM(num_l, 3, 3, activation='relu', border_mode='same')(x)
        #
        # #encoded = MaxPooling2D((2, 2), border_mode='same')(x)
        #
        # # at this point the representation is (8, 4, 4) i.e. 128-dimensional
        #
        # x = LSTM(num_l, 3, 3, activation='relu', border_mode='same')(encoded)
        # #x = UpSampling2D((2, 2))(x)
        # for i in range(1, number_of_layers):
        #     num_l += layer_step
        #     x = LSTM(num_l, 3, 3, activation='relu', border_mode='same')(x)
        #     #x = UpSampling2D((2, 2))(x)
        #
        # #decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

        autoencoder = Model(input=input_img, output=decoded)

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.encoder = autoencoder
        self.decoder = autoencoder

        x_train = list()
        for i in train_set:
            img = self.image_to_blocks(i, block_size, i.shape)
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

        autoencoder.fit(x_train, x_train,
                nb_epoch=epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))

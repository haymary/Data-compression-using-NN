from CompressionModel import CompressionModel

from keras.layers import Input, Dense, LSTM, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np

from keras.datasets import mnist

class LSTMAutoencoder(CompressionModel):
    def __init__(self, train_set, block_size, layer_step, number_of_layers, epoch, batch_size, split_rate):

        input_img = Input(shape=(1, block_size, block_size))
        num_l = block_size - layer_step

        x = LSTM(num_l, 3, 3, activation='relu', border_mode='same')(input_img)

        for i in range(1, number_of_layers):
            num_l -= layer_step
            #x = MaxPooling2D((2, 2), border_mode='same')(x)
            x = LSTM(num_l, 3, 3, activation='relu', border_mode='same')(x)

        #encoded = MaxPooling2D((2, 2), border_mode='same')(x)

        # at this point the representation is (8, 4, 4) i.e. 128-dimensional

        x = LSTM(num_l, 3, 3, activation='relu', border_mode='same')(encoded)
        #x = UpSampling2D((2, 2))(x)
        for i in range(1, number_of_layers):
            num_l += layer_step
            x = LSTM(num_l, 3, 3, activation='relu', border_mode='same')(x)
            #x = UpSampling2D((2, 2))(x)

        #decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

        autoencoder = Model(input=input_img, output=decoded)

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        (x_train, _), (x_test, _) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = np.reshape(x_train, (len(x_train), 1, block_size, block_size))
        x_test = np.reshape(x_test, (len(x_test), 1, block_size, block_size))

        autoencoder.fit(x_train, x_train,
                nb_epoch=epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))

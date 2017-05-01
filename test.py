from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

timesteps = 9
input_dim = 9
latent_dim = 5

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
# decoder = Model(encoded, decoded)
sequence_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# x_train = zeros(5)*6
# x_test = zeros(6)*4
#
# autoencoder.fit(x_train, x_train,
#                 nb_epoch=epoch,
#                 batch_size=batch_size,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))

class BlockAutoencoder():
    #BEGIN VARS
    encoder = ""
    decoder = ""
    encoding_dim = 0
    image_shape = ()
    input_image_size = 0
    block_size = 0
    #END VARS

    def train(self, train_set, block_size, layer_step, number_of_layers, epoch, batch_size, split_rate):
        self.init_values(x_train, block_size, layer_step, number_of_layers)

        input_img = Input(shape=(self.input_image_size,))
        num_l = self.input_image_size - layer_step
        encoded = Dense(num_l, activation='relu')(input_img) #23

        for _ in range(1, number_of_layers): #21 19 17
            num_l -= layer_step
            encoded = Dense(num_l, activation='relu')(encoded)

        num_l += layer_step
        decoded = Dense(num_l, activation='relu')(encoded) #19

        for _ in range(0, number_of_layers-1): #21 23 25
            num_l += layer_step
            decoded = Dense(num_l, activation='relu')(decoded)

        autoencoder = Model(input=input_img, output=decoded)

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.encoder = autoencoder
        self.decoder = autoencoder

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

    def encode(self, img):
        img_for_enc = img.astype('float32') / 255.0
        img_for_enc = np.reshape(img_for_enc, (1, len(img_for_enc)))
        return self.encoder.predict(g)

    def decode(self, encoded_imgs):
        return self.decoder.predict(i)

    def save_models(self, name):
        self.decoder.save("models/" + name + ".h5")
        self.encoder.save("models/" + name + ".h5")
        f = open('models/headers_' + name + '.txt', 'w')
        f.write(str(self.block_size) + "\n")
        f.close()

    def load_models(self, name):
        self.decoder = load_model("models/" + name + ".h5")
        self.encoder = load_model("models/" + name + ".h5")
        f = open('models/headers_' + name + '.txt', 'r')
        for line in f:
            data = line.split()
            self.block_size = int(data[0])

    def init_values(self, train_set, block_size, layer_step, number_of_layers):
        self.encoding_dim = block_size * block_size - ((number_of_layers) * layer_step)
        self.input_image_size = block_size * block_size
        self.block_size = block_size

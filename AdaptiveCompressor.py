from ..CompressionModels import LSTMAutoencoder

class AdaptiveEncoder(Compressor):
    def compress(self, image):
        box_size = 5
        im = image_to_blocks(image, box_size, image.shape)
        auto = SimpleAutoencoder()
        auto.load_models(num = str(model_number))

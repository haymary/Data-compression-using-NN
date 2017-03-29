from Compressor import Compressor
from SimpleAutoencoder import SimpleAutoencoder

class AutoencoderCompressor(Compressor):
    def compress(self, image):
        auto = SimpleAutoencoder()
        auto.load_models(num = str(num_try))

        return auto.encode(image)

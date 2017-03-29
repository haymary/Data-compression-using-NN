from Compressor import Compressor
from SimpleAutoencoder import SimpleAutoencoder

class AutoencoderCompressor(Compressor):
    def compress(self, image):
        im = self.preprocess(image)
        auto = SimpleAutoencoder()
        auto.load_models(num = str(num_try))
        return auto.encode(im)

    def preprocess(self, image):
        #TODO: do some
        image = image / 255.

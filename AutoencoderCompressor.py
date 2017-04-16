from Compressor import Compressor
from AdaptiveAutoencoder import AdaptiveAutoencoder

class AutoencoderCompressor(Compressor):
    def compress(self, image):
        model_number = 1
        im = self.preprocess(image)
        auto = AdaptiveAutoencoder()
        auto.load_models(num = str(model_number))
        return auto.encode(im)

    def preprocess(self, image):
        #TODO: do some

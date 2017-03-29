import abc

class Compressor:
    @abc.abstractmethod
    def compress(self, image):
        pass

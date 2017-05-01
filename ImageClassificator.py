
import numpy as np
from skimage.segmentation import felzenszwalb
from working_with_images import blocks_to_image

class ImageClassificator():
    def getImageClass(self, flat_img):
        img = self.make_img(flat_img)

        segments_fz = felzenszwalb(img, scale=5, sigma=0.5, min_size=1)
        num_segments  = len(np.unique(segments_fz))

        if(num_segments < 4):
            return 0
        if(num_segments < 6):
            return 1
        else:
            return 2

    def make_img(self, flat_img):
        b = list()
        b.append(flat_img)
        return blocks_to_image(b, tuple([5, 5]), 5)

import abc

class Compressor:
    @abc.abstractmethod
    def compress(self, image):
        pass

    @abc.abstractmethod
    def preprocess(self):
        pass

    def image_to_blocks(self, img, box_size, image_shape):
        x_n_boxes = int(math.ceil(image_shape[1] / float(box_size) ))
        y_n_boxes = int(math.ceil(image_shape[0] / float(box_size) ))

        result = np.zeros((x_n_boxes*y_n_boxes, box_size * box_size))

        for i in xrange(image_shape[0]):
            for j in xrange(image_shape[1]):
                x = j % box_size
                y = i % box_size
                el_num = int(x + box_size * y)
                block_num = int(math.floor(j/box_size) + (x_n_boxes * math.floor(i/box_size)))
                result[block_num,el_num] = img[i][j]
        return result

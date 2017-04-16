from keras.models import load_model
from ImageClassificator import ImageClassificator

class AdaptiveAutoencoder():
    image_shape = ()

    enc_simp = ""
    enc_med = ""
    enc_hard = ""
    dec_simp = ""
    dec_med = ""
    dec_hard = ""

    classificatior = ImageClassificator()

    def __init__(self):
        load_models()

    def encode(self, img):
        encoded = list()
        how_encoded = list()

        self.image_shape = img.shape

        blocks = self.image_to_blocks(img, self.block_size, img.shape)
        for i in range(0, len(blocks)):
            img_to_enc = blocks[i]
            im_lable = classificator.getImageClass(blocks[i])
            if (im_lable == 0):
                encoded.append(self.enc_simp.encode(img_to_enc))
            else if (im_lable == 1):
                encoded.append(self.enc_med.encode(img_to_enc))
            else:
                encoded.append(self.enc_hard.encode(img_to_enc))
            how_encoded.append(im_lable)
        encoded.append(how_encoded)

    def decode(self, img):
        encoded = img[:len(img) - 1]
        how_encoded = img[len(img) - 1]

        blocks = list()
        for i in range(0, len(encoded)):
            enc_img = encoded[i]
            if how_encoded[i] == 0:
                blocks.append(self.dec_simp.decode(enc_img))
            else if how_encoded[i] == 1:
                blocks.append(self.dec_med.decode(enc_img))
            else:
                blocks.append(self.dec_hard.decode(enc_img))
        image = self.blocks_to_image(blocks, self.image_shape, self.block_size)
        return image

    def load_models(self):
        num = ""
        enc_simp = BlockAutoencoder()
        enc_med = BlockAutoencoder()
        enc_hard = BlockAutoencoder()
        dec_simp = BlockAutoencoder()
        dec_med = BlockAutoencoder()
        dec_hard = BlockAutoencoder()

        enc_simp.load_models("enc_simp" + num)
        enc_med.load_models("enc_med" + num)
        enc_hard.load_models("enc_hard" + num)
        dec_simp.load_models("dec_simp" + num)
        dec_med.load_models("dec_med" + num)
        dec_hard.load_models("dec_hard" + num)

    def blocks_to_image(self, blocks, image_shape, box_size):
        x, y = 0, 0
        image = []
        row = []
        for i in range(0, len(blocks)):
            box = self.to_square(blocks[i], box_size)
            if(len(row) == 0):
                row = box
                row = np.array(row)
            else:
                row = self.add_box(row, box)
            if(row.shape[1] >= image_shape[1]):
                if(len(image) == 0):
                    image = row[0 : image_shape[0], 0 : image_shape[1]]
                else:
                    image = np.vstack((image, row[0 : image_shape[0] - image.shape[0], 0 : image_shape[1]]))
                row = []
        return image

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

    def to_square(self, matrix, box_size):
        return np.reshape(matrix, (box_size, box_size))

    def add_box(self, row, box):
        new_row = list()
        for i in range(0, row.shape[0]):
            new_row.append(np.append(row[i], box[i]))
        new_row = np.array(new_row)
        return new_row

from ImageClassificator import ImageClassificator
from BlockAutoencoder import BlockAutoencoder
from working_with_images import blocks_to_image, image_to_blocks
from scipy import ndimage, misc

simple = BlockAutoencoder()
medium = BlockAutoencoder()
hard = BlockAutoencoder()

# Load data
import glob
image_list = []
for filename in glob.glob('pics/sample_sm/*.jpg'):
	print "Loading {}".format(filename)
	im = misc.imread(filename, flatten=True, mode = "L")
	image_list.append(im)
image_list =  tuple(image_list)

x_simple = list()
x_medium = list()
x_hard = list()
classificator = ImageClassificator()
for img in image_list:
	img = img / 255.
	blocks = image_to_blocks(img, 5, img.shape)
	for block in blocks:
		img_class = classificator.getImageClass(block)
		if img_class == 0:
			x_simple.append(block)
		else:
			if img_class == 1:
				x_medium.append(block)
			else:
				x_hard.append(block)

simple.train(x_simple, 5, 1, 5, 100, 200, 0.3)
medium.train(x_medium, 5, 1, 5, 100, 200, 0.3)
hard.train(x_hard, 5, 1, 5, 100, 200, 0.3)
# train_set, block_size, layer_step, number_of_layers, epoch, batch_size, split_rate
simple.save_models('simple')
medium.save_models('medium')
hard.save_models('hard')

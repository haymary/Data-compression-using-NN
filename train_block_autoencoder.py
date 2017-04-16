from BlockAutoencoder import BlockAutoencoder

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

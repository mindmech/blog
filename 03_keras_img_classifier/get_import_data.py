'''
Source: mindmech.net

This script takes, as input, a folder containing multiple folders,
each labeled with a class name, and containing all the images (.png,
.jpg, or .jpeg format) pertaining to that class.

The second argument this script takes is the percentage of the data
to (randomly) separate as training data, the rest being test data.

The script will then convert all images in each class into 32x32
squares, extracting the bottom-center of the image if the image is 
not a square. So, use square images for best results!

The output file structure will be stored in the folder "images", 
where you can find the train and test data, each with all the class
folders.

Ex:
python get_import_data.py mypics 0.9
'''

import os, sys
import numpy as np
from PIL import Image
from resizeimage import resizeimage
import functions as funcs

input_dir = sys.argv[1]
image_dir = 'images'

funcs.delete_folder(image_dir)
train_perc = float(sys.argv[2])

for class_name in os.listdir(input_dir):
	for fname in os.listdir(input_dir + '/' + class_name):
		extension = fname[-3:].lower()
		if extension != 'png' and extension != 'jpg' and extension != 'jpeg':
			continue
			
		split = ''
		
		# uses train_perc as a probability, e.g. 0.9 split is seen as a 0.9 probability
		# to place the item into the train folder
		choice = np.random.choice([True, False], 1, p=[train_perc, 1 - train_perc])[0]
		if choice:
			split = 'train'
		else:
			split = 'test'
	
		img_f = open(input_dir + '/' + class_name + '/' + fname, 'rb')
		img = Image.open(img_f)
		img = resizeimage.resize_cover(img, [32, 32])
		
		new_file = image_dir + '/' + split + '/' + class_name + '/' + fname
		funcs.mkdir(new_file)
		img.save(new_file, img.format)
		img_f.close()
		
print("Imported data from", input_dir, ".")

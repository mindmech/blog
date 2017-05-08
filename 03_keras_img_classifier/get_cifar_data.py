'''
Source: mindmech.net

This script imports the CIFAR10 dataset from the keras module as
a Numpy ndarray, then converts the array into the actual RGB 
images and saves them to a directory.

The goal here is to create a file structure which can later be
read by a neural network training script, and can be replaced
interchangeably with other pictures.
'''

import functions as funcs
import os
import numpy as np
from keras.datasets import cifar10
from PIL import Image

image_dir = 'images'
funcs.delete_folder(image_dir)

# Shuffled CIFAR10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
img_idx = 0
for pair in [(x_train, y_train, 'train'), (x_test, y_test, 'test')]:
	x = pair[0]
	y = pair[1]
	
	for i in range(len(x)):
		sample_x = x[i]
		
		reshaped = np.reshape(sample_x, (32, 32, 3))
		
		pixels_out = []
		for row in reshaped:
			for item in row:
				pixels_out.append(tuple(item))
		
		image_out = Image.new("RGB",(32, 32))
		image_out.putdata(pixels_out)
		fname = image_dir + '/' + pair[2] + '/' + labels[int(y[i])] + '/' + str(img_idx) + '.png'
		funcs.mkdir(fname)
		image_out.save(fname)
		img_idx += 1
		
print("Imported data from CIFAR.")

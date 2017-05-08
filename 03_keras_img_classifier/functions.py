'''
Source: mindmech.net
'''

import pathlib, os, random
from PIL import Image
import numpy as np


'''
Some helper functions for file manipulation
'''
def delete_path(path):
	for subdir in path.iterdir():
		if subdir.is_dir():
			delete_path(subdir)
		else:
			subdir.unlink()
	path.rmdir() 

def delete_folder(dir):
	if os.path.isdir(dir):
		print("Deleting old folder: \"" + dir + "\"...")
		path = pathlib.Path(dir)
		delete_path(path)


def mkdir(fname):
	dir = os.path.dirname(fname)
	if not os.path.exists(dir):
		print("Creating folder:", dir)
		os.makedirs(dir)
		
'''
Takes an image filename and reads it into a Numpy matrix.
'''
def read_img(img_fname):
	img = Image.open(img_fname)
	pix = img.load()
	size = img.size
	#reshaped = np.reshape(pix, (3, 32, 32))
	img_x = []
	
	for row in range(size[0]):
		row_x = []
		
		for col in range(size[1]):
			row_x.append(pix[row, col])
			
		img_x.append(row_x)
	
	return np.asarray(img_x)
	

'''
Gets all train and test images from the images directory. 
Returns them as matrices, along with the labels.

This assumes the scripts for formatting the images directory 
have already been called.
'''
def get_train_test():
	result = []
	class_dict = {}
	
	for dset_name in ('train', 'test'):
		dset = []
		
		for class_name in os.listdir('images/' + dset_name + '/'):
			for fname in os.listdir('images/' + dset_name + '/' + class_name):
				extension = fname[-3:].lower()
				if extension != 'png' and extension != 'jpg' and extension != 'jpeg':
					continue
				
				full_fname = 'images/' + dset_name + '/' + class_name + '/' + fname
				x = read_img(full_fname)
				y = class_name
				dset.append((x, y))
				
		random.shuffle(dset)
		
		dset_x = [pair[0] for pair in dset]
		dset_labels = [pair[1] for pair in dset]
		
		if dset_name == 'train':
			label_idx = 0
			for label in set(dset_labels):
				class_dict[label] = label_idx
				label_idx += 1
		dset_y = []
		for label in dset_labels:
			dset_y.append(class_dict[label])
		
		result.append((np.asarray(dset_x), np.asarray(dset_y)))
	
	return result

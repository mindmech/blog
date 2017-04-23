'''
Source: http://mindmech.net
'''

import csv
import numpy as np

def process_msg(message, vocab):
	'''
	message:	the message string to classify.
	vocab: 		a dict of unique integers assigned to unique words.
	
	Insert your preprocessing here. For now we'll just lowercase, 
	skip punctuation, and add unk tags.
	'''
	msg_arr = []
	tokenized = "".join((char if char.isalpha() else " ") for char in message.lower()).split()
	
	for word in tokenized:
		
		if word in vocab:
			msg_arr.append(vocab[word])
		else:
			msg_arr.append(vocab['<unk>'])
			
	return np.asarray(msg_arr)
	

def get_vocab(train_fname):
	'''
	Creates a vocabulary from a CSV file (must have "message" column), by 
	assigning a unique integer to each unique word seen in the file. 
	Replaces words only occurring once with an <unk> tag, to give the 
	network the capability to process unknown words.
	'''
	print("Reading vocab from:", train_fname)
	reader = csv.reader(open(train_fname, 'r', encoding='utf-8'))
	freqs = {}
	
	header = next(reader)
	for row in reader:
		if row == []:
			continue
		message = row[header.index('message')]
		msg_arr = message.lower().split()
		
		for word in msg_arr:
			if word not in freqs.keys():
				freqs[word] = 0
		freqs[word] += 1
		
	vocab = {}
	vocab_idx = 1
	for word in freqs.keys():
		if freqs[word] > 1:
			vocab[word] = vocab_idx
			vocab_idx += 1
			
	vocab['<unk>'] = vocab_idx
	
	return vocab
	

def get_xy(csv_fname, vocab):
	'''
	csv_fname: 	filename for a CSV with columns "message" (string) 
				and "annotation" (int).
	vocab: 		a dict of unique integers assigned to unique words
	
	Returns "x" and "y" data from csv file, i.e. converts each message 
	into a list of corresponding word integers from the vocabulary for 
	"x". The "y" data, of course, is simply the annotation for each 
	message in the csv file.
	'''
	print("Getting x and y data from file", csv_fname)
	reader = csv.reader(open(csv_fname, 'r', encoding='utf-8'))
	header = next(reader)
	
	x = []
	y = []
	
	for row in reader:
		if row == []:
			continue
		message = row[header.index('message')]
		msg_x = process_msg(message, vocab)
		x.append(msg_x)
		
		annotation = int(row[header.index('annotation')])
		y.append(annotation)
	
	return np.asarray(x), np.asarray(y)
	
	
def load_data(train_fname, test_fname):
	'''
	Load the messages and annotations from the input CSV files as
	lists of integers assigned to vocabulary words. Return also the 
	vocabulary for later use by the live tool.
	'''
	vocab = get_vocab(train_fname)
	(x_train, y_train) = get_xy(train_fname, vocab)
	(x_test, y_test) = get_xy(test_fname, vocab)
	
	return (x_train, y_train), (x_test, y_test), vocab
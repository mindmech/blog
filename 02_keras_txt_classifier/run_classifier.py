'''
Source: http://mindmech.net
'''

from keras.models import load_model
from keras.preprocessing import sequence
import data_functions as funcs
import pickle
import numpy as np

model = load_model('classifier.h5')
vocab = pickle.load(open('vocab.pkl', 'rb'))

command = ''

print("Enter a message and see its sentiment:")
while True:
	command = input('-> ')
	if command == 'exit':
		break
	
	x = funcs.process_msg(command, vocab)
	x = sequence.pad_sequences([x], maxlen=400)
	print("Sentiment:", model.predict(x)[0][0])

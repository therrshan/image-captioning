from os import listdir
from pickle import dump
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import numpy as np
import string

# Image Feature Extraction
def extract_features(directory):
	model = InceptionV3(weights='imagenet')
	model = Model(inputs=model.input, outputs=model.layers[-2].output)

	features = dict()

	for name in listdir(directory):
		filename = directory + '/' + name
		img = load_img(filename, target_size=(299, 299))

		x = img_to_array(img)
		x = np.expand_dims(x, axis=0)
		
		x = preprocess_input(x)

		feature = model.predict(x, verbose=0)
		image_id = name.split('.')[0]
		features[image_id] = feature

	return features


directory = 'data/Flicker8k_Dataset'
features = extract_features(directory)
dump(features, open('features.pkl', 'wb'))

filename = 'data/Flicker8k_text/Flickr8k.token.txt'
file = open(filename, 'r')
data = file.read()

#create captions dictionary
captions = dict()
for line in data.split('\n'):
	words = line.split()
	if len(words) < 2:
		continue
	img_id, img_desc = words[0], words[1:]
	img_id = img_id.split('.')[0]
	img_desc = ' '.join(img_desc)
	if img_id not in captions:
		captions[img_id] = list()
	captions[img_id].append(img_desc)

#clean the captions
table = str.maketrans('', '', string.punctuation)
for img_id, true_captions in captions.items():
	for i in range(len(true_captions)):
		caption = true_captions[i]
		caption = caption.split()
		caption = [word.lower() for word in caption]
		caption = [w.translate(table) for w in caption] 
		caption = [word for word in caption if len(word)>1]
		caption = [word for word in caption if word.isalpha()] 
		true_captions[i] =  ' '.join(caption)

#create clean captions list
temp = list()
for img_id, true_captions in captions.items():
	for caption in true_captions:
		temp.append(img_id + ' ' + caption)
data = '\n'.join(temp)
file = open('captions.txt', 'w')
file.write(data)
file.close()
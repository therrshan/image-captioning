from os import listdir
from pickle import dump
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import numpy as np


def extract_features(directory):
	model = InceptionV3(weights='imagenet')
	model = Model(inputs=model.input, outputs=model.layers[-2].output)
	print(model.summary())
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

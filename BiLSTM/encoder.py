import os
import cv2
import numpy as np
from keras.src.applications import InceptionV3
from skimage import io
from tqdm import tqdm

model = InceptionV3(include_top=False, pooling='avg', weights='imagenet')

image_features = {}

current_working_directory = os.getcwd()
image_path = current_working_directory + "/flickr/Images/"


def get_image_features(images):
    for img in tqdm(images):
        image = io.imread(image_path + img)
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize each image size 299 x 299
        image = cv2.resize(image, (299, 299))
        image = np.expand_dims(image, axis=0)

        # Normalize image pixels
        image = image / 127.5
        image = image - 1.0

        # Extract features from image
        feature = model.predict(image)
        image_features[img] = feature
    return image_features

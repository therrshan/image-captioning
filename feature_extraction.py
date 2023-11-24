import os
import numpy as np
import pickle
import string
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Function to preprocess captions
def preprocess_caption(caption):
    caption = caption.lower()  # Convert to lowercase
    caption = caption.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(caption)
    caption = '<start> ' + ' '.join(tokens)+ ' <end>'
    return caption

# Function to preprocess images before feature extraction
def preprocess_images(image_directory, image_ids):
    processed_images = {}
    for img_id in image_ids:
        img_id = img_id + '.jpg'
        img_path = os.path.join(image_directory, img_id)
        img = Image.open(img_path)
        img = img.resize((299, 299))  # Resize image to a standard size (e.g., 299x299 for InceptionV3)
        img_array = keras_image.img_to_array(img)
        img_array /= 255.0  # Normalize pixel values to [0, 1]
        processed_images[img_id] = img_array
    return processed_images

# Function to extract image features
def extract_image_features(processed_images):
    print("Starting Feature Extraction")
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    print("Output Dimension:", model.output_shape)
    image_features = {}
    for img_id, img_array in processed_images.items():
        img_id = img_id.split('.')[0]
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        image_features[img_id] = features.flatten()
    return image_features

# Paths to caption and image ID text files
train_ids_file = 'data/Flicker8k_text/Flickr_8k.trainImages.txt'
test_ids_file = 'data/Flicker8k_text/Flickr_8k.testImages.txt'
val_ids_file = 'data/Flicker8k_text/Flickr_8k.devImages.txt'
caption_txt_path = 'data/Flicker8k_text/Flickr8k.token.txt'
image_directory = 'data/Flicker8k_Dataset'

train_ids = []
test_ids = []
val_ids = []

# Load training and testing image IDs from text files
with open(train_ids_file, 'r') as train_file:
    _train_ids = train_file.read().splitlines()
    for x in _train_ids:
        x = x.split('.')[0]
        train_ids.append(x)

with open(test_ids_file, 'r') as test_file:
    _test_ids = test_file.read().splitlines()
    for x in _test_ids:
        x = x.split('.')[0]
        test_ids.append(x)

with open(val_ids_file, 'r') as val_file:
    _val_ids = val_file.read().splitlines()
    for x in _val_ids:
        x = x.split('.')[0]
        val_ids.append(x)

# Load captions and create word mappings
def caption_processing(caption_txt_path, train_ids, test_ids, val_ids):
    print("Starting Caption Processing")
    all_captions = {}

    content = open(caption_txt_path, 'r').read()
    for line in content.split('\n'):
        tokens=line.split()
        if len(line) > 2:
            image_id = tokens[0].split('#')[0].split('.')[0]
            image_desc = ' '.join(tokens[1:])
            if image_id not in all_captions:
                all_captions[image_id] = list()
            all_captions[image_id].append(image_desc)

    caption_vocab = []

    for img_id in all_captions:
        for caption in all_captions[img_id]:
            caption_vocab.append(caption)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(caption_vocab)
    vocab_size = len(tokenizer.word_index) + 1


    # Extract captions for training and testing images
    train_captions = {}
    test_captions = {}
    val_captions = {}

    for img_id in all_captions:
        if img_id in train_ids:
            train_captions[img_id] = [preprocess_caption(c) for c in all_captions[img_id]]
        elif img_id in test_ids:
            test_captions[img_id] = [preprocess_caption(c) for c in all_captions[img_id]]
        elif img_id in val_ids:
            val_captions[img_id] = [preprocess_caption(c) for c in all_captions[img_id]]

    # # Create word-to-index and index-to-word mappings
    # wordtoix = {}
    # ixtoword = {}
    # for img_captions in all_captions.values():
    #     for caption in img_captions:
    #         tokens = word_tokenize(caption)
    #         for token in tokens:
    #             if token not in wordtoix:
    #                 idx = len(wordtoix) + 1  # Index starts from 1 (reserve 0 for padding)
    #                 wordtoix[token] = idx
    #                 ixtoword[idx] = token
    return train_captions, test_captions, val_captions, caption_vocab

# Preprocess Captions
train_captions, test_captions, val_captions, caption_vocab = caption_processing(caption_txt_path, train_ids, test_ids, val_ids)

# Preprocess images for training and testing sets
train_processed_images = preprocess_images(image_directory, train_ids)
test_processed_images = preprocess_images(image_directory, test_ids)
val_processed_images = preprocess_images(image_directory, val_ids)

image_ids = train_processed_images.copy()
image_ids.update(test_processed_images)
image_ids.update(val_processed_images)

# Extract features for training and testing images
train_features = extract_image_features(train_processed_images)
test_features = extract_image_features(test_processed_images)
val_features = extract_image_features(val_processed_images)

print("Creating Pickle files")
# Save features, captions, and word mappings as pickle files
with open('train_features.pkl', 'wb') as f:
    pickle.dump(train_features, f)

with open('test_features.pkl', 'wb') as f:
    pickle.dump(test_features, f)

with open('val_features.pkl', 'wb') as f:
    pickle.dump(val_features, f)

with open('train_captions.pkl', 'wb') as f:
    pickle.dump(train_captions, f)

with open('test_captions.pkl', 'wb') as f:
    pickle.dump(test_captions, f)

with open('val_captions.pkl', 'wb') as f:
    pickle.dump(val_captions, f)

with open('caption_vocab.pkl', 'wb') as f:
    pickle.dump(caption_vocab, f)

with open('train_ids.pkl', 'wb') as f:
    pickle.dump(train_ids, f)

with open('test_ids.pkl', 'wb') as f:
    pickle.dump(test_ids, f)
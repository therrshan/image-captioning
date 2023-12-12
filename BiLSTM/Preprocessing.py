import os
import pickle
import string
import warnings

from keras.preprocessing.sequence import pad_sequences
from keras.src.preprocessing.text import Tokenizer
from itertools import chain
import numpy as np
import random

import encoder

warnings.filterwarnings("ignore")
current_working_directory = os.getcwd()
image_path = current_working_directory + "/flickr/Images"
annotations_file = current_working_directory + "/flickr/captions.txt"

jpgs = os.listdir(image_path)

print("Total Images in Dataset = {}".format(len(jpgs)))

dataset = dict()
max_length_caption = 0
captions = open(annotations_file, 'r', encoding="utf8").read().split("\n")
captions = captions[1:]
for read_line in captions:
    caption_col_data = read_line.split(',')
    if len(caption_col_data) <= 1:
        break
    w = caption_col_data[0].split("#")
    cap = caption_col_data[1].translate(str.maketrans('', '', string.punctuation))
    # Replace - to blank
    cap = cap.replace("-", " ")

    # Split string into word list and Convert each word into lower case
    cap = cap.split()
    cap = [word.lower() for word in cap]

    # join word list into sentence and <start> and <end> tag to each sentence which helps
    # LSTM encoder-decoder model while training.
    cap = '<start> ' + " ".join(cap) + ' <end>'
    if w[0] not in dataset.keys():
        dataset[w[0]] = []
    max_length_caption = max(max_length_caption, len(cap.split()))
    dataset[w[0]].append(cap.lower())

print(max_length_caption)
print("Length of Dataset: ", len(dataset))

flatten_list = list(chain.from_iterable(dataset.values()))  # [[1,3],[4,8]] = [1,3,4,8]
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', oov_token='<unk>')  # For those
# words which are
# not found in word_index
tokenizer.fit_on_texts(flatten_list)
total_words = len(tokenizer.word_index) + 1

print("Vocabulary length: ", total_words)
print("Bicycle ID: ", tokenizer.word_index['bicycle'])
print("Airplane ID: ", tokenizer.word_index['airplane'])
print('<start>:', tokenizer.word_index['<start>'])
print('<end>:', tokenizer.word_index['<end>'])

# for getting image features using InceptionV3 CNN model

# image_features = encoder.get_image_features(dataset.keys())
# print("Image features length: ", len(image_features))

# with open("encoded_train_images_inceptionV3.p", "wb") as encoded_pickle:
#     pickle.dump(image_features, encoded_pickle)

img_encodings = pickle.load(open('encoded_train_images_inceptionV3.p', 'rb'))

word_idx = {val: index for index, val in enumerate(tokenizer.word_index)}
idx_word = {index: val for index, val in enumerate(tokenizer.word_index)}


def split_dict(dictionary, n):
    # Get the keys from the dictionary
    keys = list(dictionary.keys())

    # Get the first n random values
    a_keys = random.sample(keys, n)
    a = {key: dictionary[key] for key in a_keys}

    # Get the rest (n-1) values
    b_keys = [key for key in keys if key not in a_keys]
    b = {key: dictionary[key] for key in b_keys}

    return a, b


training_dataset, test_dataset = split_dict(dataset, 7091)
captionz = []
img_id = []

for img in training_dataset.keys():
    for caption in training_dataset[img]:
        captionz.append(caption)
        img_id.append(img)

print(len(captionz), len(img_id))
no_samples = 0
for caption in captionz:
    no_samples += len(caption.split()) - 1
print(no_samples)

caption_length = [len(caption.split()) for caption in captionz]
max_length_caption = max(caption_length)


def data_process(batch_size):
    partial_captions = []
    next_words = []
    images = []
    total_count = 0
    while 1:

        for image_counter, caption in enumerate(captionz):
            current_image = img_encodings[img_id[image_counter]][0]
            for i in range(len(caption.split()) - 1):
                total_count += 1
                partial = [word_idx[txt] for txt in caption.split()[:i + 1]]
                partial_captions.append(partial)
                next = np.zeros(total_words)
                next[word_idx[caption.split()[i + 1]]] = 1
                next_words.append(next)
                images.append(current_image)

                if total_count >= batch_size:
                    next_words = np.asarray(next_words)
                    images = np.asarray(images)
                    partial_captions = pad_sequences(partial_captions, maxlen=max_length_caption, padding='post')
                    total_count = 0
                    yield [[images, partial_captions], next_words]
                    partial_captions = []
                    next_words = []
                    images = []

import numpy as np
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import string


def get_image_ids(filename):
    file = open(filename, 'r')
    text = file.read()
    img_ids = list()
    for line in text.split('\n'):
        img_id = line.split('.')[0]
        img_ids.append(img_id)
    return set(img_ids)

def get_image_captions(filename, dataset):
    file = open(filename, 'r')
    text = file.read()
    captions = dict()
    for line in text.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        
        if image_id in dataset:
            if image_id not in captions:
                captions[image_id] = list()
            caption = 'startseq ' + ' '.join(image_desc) + ' endseq'

            captions[image_id].append(caption)
    return captions

def get_image_features(filename, dataset):
    all_features = load(open(filename, 'rb'))

    features = {k: all_features[k] for k in dataset}
    return features

def create_tokenizer(captions):
    descriptions = list()
    for img_id in captions.keys():
        [descriptions.append(d) for d in captions[img_id]]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(descriptions)

    vocab_size = len(tokenizer.word_index) + 1
    max_len = max(len(d.split()) for d in descriptions) 

    return tokenizer, vocab_size, max_len

def create_sequences(tokenizer, max_length, true_captions, image, vocab_size):

    """
    Create sequences for training the model.

    Parameters:
    - tokenizer (Tokenizer): Tokenizer object used to convert text to sequences.
    - max_length (int): Maximum length of input sequences.
    - true_captions (list): List of image captions.
    - image (array): Feature vector representing the image.
    - vocab_size (int): Size of the vocabulary.

    Returns:
    - array: Numpy array containing input data (image), input sequences (in_seq), and output sequences (out_seq).
    """

    X1, X2, y = list(), list(), list()
    for caption in true_captions:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post', truncating='post')[0]

            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        
            X1.append(image)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)

def data_generator(captions, image_features, tokenizer, max_length, vocab_size):

    """
    Data Generator function

    Parameters:
    - captions (dict): Dictionary containing image IDs as keys and corresponding lists of captions.
    - image_features (dict): Dictionary containing image IDs as keys and corresponding feature vectors.
    - tokenizer (Tokenizer): Tokenizer object used to convert text to sequences.
    - max_length (int): Maximum length of input sequences.
    - vocab_size (int): Size of the vocabulary.

    Yields:
    - tuple: Tuple containing input data (image and sequence) and output data (word).
    """


    while 1:
        for img_id, true_captions in captions.items():
            image = image_features[img_id][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, true_captions, image, vocab_size)
            yield [[in_img, in_seq], out_word]
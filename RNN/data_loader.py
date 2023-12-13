import numpy as np
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import string

def load_captions(file):
    mapping = dict()
    for line in file.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping

def clean_captions(captions):
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

def to_vocabulary(captions):
    all_desc = set()
    for img_id in captions.keys():
        [all_desc.update(d.split()) for d in captions[img_id]]
    return all_desc

def save_captions(captions, filename):
    lines = list()
    for img_id, true_captions in captions.items():
        for caption in true_captions:
            lines.append(img_id + ' ' + caption)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def get_image_ids(filename):
    file = open(filename, 'r')
    text = file.read()
    dataset = list()
    for line in text.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def get_clean_captions(filename, dataset):
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

def to_lines(captions):
    all_desc = list()
    for img_id in captions.keys():
        [all_desc.append(d) for d in captions[img_id]]
    return all_desc

def create_tokenizer(captions):
    lines = to_lines(captions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(captions):
    lines = to_lines(captions)
    return max(len(d.split()) for d in lines)

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
            


filename = 'data/Flicker8k_text/Flickr8k.token.txt'
file = open(filename, 'r')
text = file.read()

captions = load_captions(text)

clean_captions(captions)
save_captions(captions, 'captions.txt')

vocabulary = to_vocabulary(captions)
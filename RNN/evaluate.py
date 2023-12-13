from numpy import argmax, log
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import data_loader

def get_word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_description(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post', truncating='post')
        pred = model.predict([photo, sequence], verbose=0)
        pred = argmax(pred)
        word = get_word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def evaluate_model_bleu_scores(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    for key, desc_list in descriptions.items():
        pred = generate_description(model, tokenizer, photos[key], max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(pred.split())

    print('CORPUS - BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1, 0, 0, 0)))
    print('CORPUS - BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('CORPUS - BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('CORPUS - BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

filename_train = 'data/Flicker8k_text/Flickr_8k.trainImages.txt'
train_image_ids = data_loader.get_image_ids(filename_train)
train_descriptions = data_loader.get_image_captions('descriptions.txt', train_image_ids)
tokenizer, vocab_size, max_caption_length = data_loader.create_tokenizer(train_descriptions)

filename_test = 'data/Flicker8k_text/Flickr_8k.testImages.txt'
test_image_ids = data_loader.get_image_ids(filename_test)
test_descriptions = data_loader.get_image_captions('descriptions.txt', test_image_ids)
test_image_features = data_loader.get_image_features('features.pkl', test_image_ids)

filename_model = 'models/model_19.h5'
trained_model = load_model(filename_model)

evaluate_model_bleu_scores(trained_model, test_descriptions, test_image_features, tokenizer, max_caption_length)

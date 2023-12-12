import os

from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
import Preprocessing
import models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


# for getting image features of validation images
def predict_captions(img_file):
    start_word = ["<start>"]
    while 1:
        now_caps = [Preprocessing.word_idx[i] for i in start_word]
        now_caps = pad_sequences([now_caps], maxlen=Preprocessing.max_length_caption, padding='post')
        e = Preprocessing.img_encodings[img_file][0]
        predictions = models.fin_model.predict([np.array([e]), np.array(now_caps)])
        word_pred = Preprocessing.idx_word[np.argmax(predictions[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > Preprocessing.max_length_caption:
            # keep on predicting next word until word predicted is <end> or caption lengths is greater than
            # max_length(40)
            break

    return ' '.join(start_word[1:-1])


def beam_search_predictions(img_file, beam_index=3):
    start = [Preprocessing.word_idx["<start>"]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < Preprocessing.max_length_caption:
        temp = []
        for s in start_word:
            now_caps = pad_sequences([s[0]], maxlen=Preprocessing.max_length_caption, padding='post')
            e = Preprocessing.img_encodings[img_file][0]
            predictions = models.fin_model.predict([np.array([e]), np.array(now_caps)])

            word_predictions = np.argsort(predictions[0])[-beam_index:]

            # Getting the top Beam index = 3  predictions and creating a
            # new list to put them via the model again
            for w in word_predictions:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += predictions[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [Preprocessing.idx_word[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


test_dataset = Preprocessing.test_dataset
current_working_directory = os.getcwd()
image_path = current_working_directory + "/flickr/Images/"

for i in range(5):
    image_file = list(test_dataset.keys())[random.randint(0, 1000)]
    test_image = image_path + str(image_file)
    # Read the image using matplotlib
    image = mpimg.imread(test_image)

    caption1 = predict_captions(image_file)
    caption2 = beam_search_predictions(image_file, beam_index=3)
    caption3 = beam_search_predictions(image_file, beam_index=5)

    actual, predicted, beam_predicted, beam_predicted2 = list(), list(), list(), list()
    references = [d.split()[1:-1] for d in test_dataset[image_file]]
    actual.append(references)
    predicted.append(caption1.split())
    beam_predicted.append(caption2.split())
    beam_predicted2.append(caption3.split())
    bleu = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_beam = corpus_bleu(actual, beam_predicted, weights=(0.5, 0.5, 0, 0))
    bleu_beam2 = corpus_bleu(actual, beam_predicted2, weights=(0.5, 0.5, 0, 0))

    # Display the image
    plt.imshow(image)
    plt.title("Your Image Title")
    plt.axis('off')  # Turn off axis labels
    plt.show()

    print('Greedy search:', caption1, 'Bleu is:', str(bleu))
    print('Beam Search, k=3:', caption2, 'Bleu is:', str(bleu_beam))
    print('Beam Search, k=5:', caption3, 'Bleu is:', str(bleu_beam2))

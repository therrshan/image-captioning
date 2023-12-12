from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import Preprocessing
import test
import tqdm

reference = Preprocessing.test_dataset
actual, predicted, beam_predicted = list(), list(), list()
for key, desc_list in (reference.items()):
    caption = test.predict_captions(key)
    beam_caption = test.beam_search_predictions(key, beam_index=5)
    references = [d.split()[1:-1] for d in desc_list]
    actual.append(references)
    predicted.append(caption.split())
    beam_predicted.append(beam_caption.split())

print("Greedy Search Predicted Captions BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print("Beam Search K=5 Predicted Captions BLEU-2: %f" % corpus_bleu(actual, beam_predicted, weights=(0.5, 0.5, 0, 0)))

print('CORPUS - BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1, 0, 0, 0)))
print('CORPUS - BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('CORPUS - BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
print('CORPUS - BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


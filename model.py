import pickle
import numpy as np
from numpy import argmax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, add, Attention, Reshape
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical, plot_model
from nltk.translate.bleu_score import corpus_bleu

# Load data
with open('train_features.pkl', 'rb') as f:
    train_image_features = pickle.load(f)

with open('train_captions.pkl', 'rb') as f:
    train_captions = pickle.load(f)

with open('test_features.pkl', 'rb') as f:
    test_image_features = pickle.load(f)

with open('test_captions.pkl', 'rb') as f:
    test_captions = pickle.load(f)

with open('val_features.pkl', 'rb') as f:
    val_image_features = pickle.load(f)

with open('val_captions.pkl', 'rb') as f:
    val_captions = pickle.load(f)

with open('caption_vocab.pkl', 'rb') as f:
    caption_vocab = pickle.load(f)

with open('train_ids.pkl', 'rb') as f:
    train_ids = pickle.load(f)

with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(caption_vocab)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in caption_vocab)

print(train_image_features['3415311628_c220a65762'].shape)

def data_generator(data_keys, features, captions, tokenizer, max_length, vocab_size, batch_size):
    while True:
        X1, X2, y = list(), list(), list()
        for key in data_keys:
            feature = features[key]
            image_captions = captions[key][:5]  

            print("Entered Data Generator")
        
            feature = np.reshape(feature, (1, feature.shape[0]))
            
            for caption in image_captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(feature) 
                    X2.append(in_seq) 
                    y.append(out_seq)  

            if len(X1) >= batch_size:
                yield [np.array(X1[:batch_size]), np.array(X2[:batch_size])], np.array(y[:batch_size])
                X1, X2, y = X1[batch_size:], X2[batch_size:], y[batch_size:]


inputs1 = Input(shape=(None, 2048))  
fe1 = LSTM(256, return_sequences=True)(inputs1)  

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = LSTM(256)(se1)


decoder1 = add([fe1, se2])
decoder2 = LSTM(256, return_sequences=False)(decoder1)  
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


model.summary()


epochs = 10
batch_size = 32
steps = len(train_ids) // batch_size

for i in range(epochs):
  
    generator = data_generator(train_ids, train_image_features, train_captions, tokenizer, max_length, vocab_size, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)\
    

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def generate_desc(model, tokenizer, image_desc, max_length):

	in_text = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length, padding='post', truncating='post')
		yhat = model.predict([image_desc,sequence], verbose=0)
		yhat = argmax(yhat)
		word = word_for_id(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	for key, desc_list in descriptions.items():
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())

	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))


evaluate_model(model, test_captions, test_image_features, tokenizer, max_length)
import data_loader, model
from pickle import dump


filename_train = 'data/Flicker8k_text/Flickr_8k.trainImages.txt'
train_image_ids = data_loader.get_image_ids(filename_train)
train_descriptions = data_loader.get_clean_captions('descriptions.txt', train_image_ids)
train_image_features = data_loader.get_image_features('features.pkl', train_image_ids)
tokenizer = data_loader.create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
max_length = data_loader.max_length(train_descriptions)

rnn = model.define_model(vocab_size, max_length)
epochs = 20
steps = len(train_descriptions)

for i in range(epochs):
	generator = data_loader.data_generator(train_descriptions, train_image_features, tokenizer, max_length, vocab_size)
	rnn.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	rnn.save('models/model_' + str(i) + '.h5')
import os
import nltk
nltk.download('punkt')
os.environ["KERAS_BACKEND"] = "tensorflow"

import re
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from keras.applications import efficientnet
from keras.layers import TextVectorization

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from train import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel

keras.utils.set_random_seed(111)



# Recreate the model architecture
cnn_model = get_cnn_model()  # Assuming get_cnn_model is your function to create the CNN part
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    image_aug=image_augmentation  # Assuming image_augmentation is defined
)

# Load the weights
caption_model.load_weights('/content/model_weights_inception')



vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1



def generate_caption(sample_img):


    # Read the image from the disk
    sample_img = decode_and_resize(sample_img)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    # plt.imshow(img)
    # plt.show()

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        # print(predictions)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    # print("Predicted Caption: ", decoded_caption)
    return decoded_caption



def evaluate_model_on_some_images(model, test_data, vectorization):
    bleu_scores = []
    count=0
    arr=['3385593926_d3e9c21170.jpg','2677656448_6b7e7702af.jpg',
'311146855_0b65fdb169.jpg',
'1258913059_07c613f7ff.jpg',
'241347760_d44c8d3a01.jpg',
'2654514044_a70a6e2c21.jpg',
'2339106348_2df90aa6a9.jpg',
'256085101_2c2617c5d0.jpg',
'280706862_14c30d734a.jpg',
'3072172967_630e9c69d0.jpg',
'3482062809_3b694322c4.jpg',
'1167669558_87a8a467d6.jpg',
'2847615962_c330bded6e.jpg',
'3344233740_c010378da7.jpg',
'2435685480_a79d42e564.jpg']

    arr2=[]
    for st in arr:
      arr2.append('Flicker8k_Dataset/'+st)
    for img_path, true_captions in test_data.items():


        if img_path in arr2:

          predicted_caption = generate_caption(img_path) 


          # Calculate BLEU score

          chencherry = SmoothingFunction()

          bleu_score = sentence_bleu(true_captions, predicted_caption,
                            smoothing_function=chencherry.method1)

          bleu_scores.append(bleu_score)

          sample_img = decode_and_resize(img_path)
          img = sample_img.numpy().clip(0, 255).astype(np.uint8)
          plt.imshow(img)
          plt.show()
          count+=1


    # Compute average BLEU score
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(len(bleu_scores))
    return average_bleu


bleu_score = evaluate_model_on_some_images(caption_model, test_data, vectorization)
print(f"Average BLEU Score on the some test images : {bleu_score}")



def evaluate_model_on_test_data(model, test_data, vectorization):
    bleu_scores = []
    count=0
    
    for img_path, true_captions in test_data.items():


          predicted_caption = generate_caption(img_path) # You need to define this function


          # Calculate BLEU score

          chencherry = SmoothingFunction()

          bleu_score = sentence_bleu(true_captions, predicted_caption,
                            smoothing_function=chencherry.method1)

          bleu_scores.append(bleu_score)


    # Compute average BLEU score
    average_bleu = sum(bleu_scores) / len(bleu_scores)

    return average_bleu


bleu_score = evaluate_model(caption_model, test_data, vectorization)
print(f"Average BLEU Score on the test set: {bleu_score}")

import data_loader, evaluate
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

test_filename = 'data/Flicker8k_text/Flickr_8k.testImages.txt'
test_set = data_loader.get_image_ids(test_filename)
test_descriptions = data_loader.get_clean_captions('descriptions.txt', test_set)
test_features = data_loader.get_image_features('features.pkl', test_set)


images = ['3385593926_d3e9c21170.jpg', '2677656448_6b7e7702af.jpg', '311146855_0b65fdb169.jpg',
              '1258913059_07c613f7ff.jpg', '241347760_d44c8d3a01.jpg', '2654514044_a70a6e2c21.jpg',
              '2339106348_2df90aa6a9.jpg', '256085101_2c2617c5d0.jpg', '280706862_14c30d734a.jpg',
              '3072172967_630e9c69d0.jpg', '3482062809_3b694322c4.jpg', '1167669558_87a8a467d6.jpg',
              '2847615962_c330bded6e.jpg', '3344233740_c010378da7.jpg', '2435685480_a79d42e564.jpg']

captions_preds = []
caption_actual = []

# Load the model
model_filename = 'model_9.h5'
model = load_model(model_filename)

def plot_image_caption(img_path, predicted_caption, img):

    img = mpimg.imread(img_path)
    predicted_caption = predicted_caption.replace('startseq', '').replace('endseq', '')

    plt.figure(facecolor='#e7e7e7')
    plt.imshow(img)
    plt.axis('off')
    
    plt.title(f'{predicted_caption}', fontsize=16, color='#4d4d4d', fontweight="bold", wrap=True)
    plt.tight_layout()

    plt.savefig(f'outputs/pred.png', bbox_inches='tight')

for img in images:
    photo = test_features[img.split('.')[0]]
    caption = evaluate.beam_search_generate_desc(model, data_loader.tokenizer, photo, 34)
    print(f"{img} : {caption}")
    captions_preds.append(caption.split())
    references = [d.split() for d in test_descriptions[img.split('.')[0]]]
    caption_actual.append(references)

    img_path = 'data/Flicker8k_Dataset/' + img
    plot_image_caption(img_path, caption, img)

# Calculate BLEU score
bleu_score = corpus_bleu(caption_actual, captions_preds, weights=(0.5, 0.5, 0, 0))
print('Corpus BLEU: %f' % bleu_score)
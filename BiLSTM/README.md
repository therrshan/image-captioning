Bi-LSTM Code Base Structure :
<br>**Preprocessing.py** : Does all the preprocessing of the data which includes the helper methods required to clean captions, load image features and get the image ids from the annotations file. This also feeds data to the model while training. 
<br>**bleu.py** : Evaluation script that runs on the entire test dataset and calculates the BLEU scores.
<br>**encoder.py** : Extracts the features from the images in the dataset using the InceptionV3 with pretrained weights from Imagenet.
<br>**models.py** : The model architecture.
<br>![model](results/model-caption.png)
<br>**test.py** : Has the script to test the model on single images to get the predictions.
<br>**train.py** : Trains the model.
<br>**image_captioning.py** : The consolidated code , having all the code from the above files at a single place.
<br>The **results folder** consists of few test images and their predicted captions. **Flickr folder** has the saved image features for the flickr images. The weights of the trained model can be found here - https://drive.google.com/file/d/1rKjH09jB0aBfRGnfHAesd_JXUa8Y8u3A/view?usp=sharing

RNN Code Base Structure :  

1. data_loader.py : All the helper functions required to clean captions, load image features and get the image ids from the annotations file.  
2. evaluate.py : Evaluation script that runs on the entire test dataset and calculates the BLEU scores.  
3. feature_extraction.py : Extracts the features from the images in the dataset using the InceptionV3 with pretrained weights from Imagenet.  
4. model.py : The model architecture.  
5. test.py : Has the script to test the model on single images to get the predictions.  
6. train.py : Trains the model.

The output folder consists of few test images and their predicted captions. Models has the saved model weight for the final epoch that we used to test.  


# RNN + GRU Model for Image Captioning

This repository contains the code for a transformer-based image captioning model. The model is trained to generate captions for images using a transformer architecture.

## Code Structure

### `feature_extraction.py`

This file takes the image directory and runs the InceptionV3 model to extract the image features and dump them in the pkl file. Captionn cleaning is also done in this script.

### `data_loader.py`

This file contains the necessary functions for loading the dataset, cleaning the provided image captions and getting image ids to split the data for training and testing.

### `model.py`

The model architetcure is defined in this file.

### `train.py`

In this file, the functions from dataloader are used to load the training data the model is trained.

### `evaluate.py`

This file contains the necessary functions to generate the predicted captions and evaluate the model on the test dataset

### `test.py`

This file takes in the image_ids from the test file individually to generate and display the predicted caption.

## Output

The `outputs` folder contains a selection of test images along with their corresponding predicted and true captions.

## Model Weights

The trained model weights are essential for reproducing the results and further experimentation. These could not be uploaded to github due to the file size limit. You can download the model weights from the following [link](https://drive.google.com/drive/folders/1XjiCD8myubTP5rMH38FVGbuLdLmKtr6x?usp=drive_link).

## Contributions

Contributions to this project are welcome. Please feel free to fork the repository, make your changes, and create a pull request.


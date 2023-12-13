# Transformer Model for Image Captioning

This repository contains the code for a transformer-based image captioning model. The model is trained to generate captions for images using a transformer architecture.

## Code Structure

### `train.py`

This file contains the necessary functions for loading the dataset and defining the model architecture. It includes the training loop, where the model is trained for a specified number of epochs.

### `test.py`

In this file, the trained model weights are loaded to evaluate the model's performance on test data. It outputs the predicted captions and the BLEU score.

## Output

The `output` folder contains a selection of test images along with their corresponding predicted and true captions, showcasing the model's captioning abilities.

## Model Weights

The trained model weights are essential for reproducing the results and further experimentation. You can download the model weights from the following [link](https://drive.google.com/drive/folders/1PUL22-bBybSM2iLgBmiwFQWF7sjJa_tk?usp=sharing).

## Getting Started

To get started with this project, clone this repository, download the model weights from the provided link, and place them in the appropriate directory. Then, run `train.py` to train the model or `test.py` to evaluate its performance on the test dataset.

## Contributions

Contributions to this project are welcome. Please feel free to fork the repository, make your changes, and create a pull request.



# ImageColorizationusingInceptionV3 Deep learning model

This repository contains code and resources for image colorization using deep learning. The project leverages convolutional neural networks and transfer learning to automatically colorize grayscale images. The colorization process is based on the principles of computer vision and machine learning.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Colorization](#colorization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Image colorization is the process of adding colors to black and white images. In this project, we utilize a deep learning architecture that combines an encoder-decoder network with an InceptionV3-based feature embedding. The model is trained on a dataset of grayscale images and their corresponding color images, learning to predict color information based on the provided grayscale input.

## Dependencies

Make sure you have the following dependencies installed:

- Python (>= 3.6)
- TensorFlow (>= 2.0)
- NumPy
- PIL (Pillow)
- scikit-image


## Usage

1. Clone this repository:

git clone https://github.com/Mazrah18/ImageColorizationusingInceptionV3.git
cd into the repo

## Dataset
The dataset used for training and evaluation is located in the true_images/ directory. It consists of pairs of grayscale images and their corresponding color images. You can replace or augment this dataset with your own images for specific applications.


Training
To train the colorization model, follow these steps:

1. Preprocess your training data and load it using the provided functions.

2. Adjust hyperparameters and model architecture in the training section of the code.

3. Train the model using the model.fit() function.

4. Trained model weights will be saved periodically in the weights/ directory.

Colorization
After training, you can use the trained model to colorize grayscale images. Follow these steps:

1. Place your grayscale images in the bw_images/ directory.

2. Run the colorization process using the trained model:

Inception_v3.ipynb file


# STEPS TO RUN:
Readme file to execute Inception v3 model

There are 2 ways to run the code in your system.

USING TENSORFLOW GPU(If want to use GPU to train):
Tensorflow-GPU is not available in latest versions of python so we can create a virtual environment.

I have used the anaconda tool to run my code. In order to reproduce the same configuration(Windows 11):

1.	Install Anaconda in your system.
2.	Open Anaconda Prompt.
3.	conda create --name <any_env_name> python=3.9
4.	You can deactivate and activate it with the following commands.
a.	conda deactivate
b.	conda activate <any_env_name>
5.	Then install the CUDA, cuDNN with conda.
a.	conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
6.	pip install - - upgrade pip
Anything above 2.10 is not supported on the GPU on Windows Native
7.	pip install "tensorflow<2.11"
8.	Check your installation of GPU using the command
a.	python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
9.	Next, follow the steps in the next section(Without GPU)

WITHOUT GPU:
1.	Navigate to the directory with the file.
2.	Install the necessary import libraries into your system using pip(Note: for users using GPU, you need to install the libraries in the anaconda prompt instead of normal terminal after activating the virtual environment).
3.	Run the cells in the ipynb file.

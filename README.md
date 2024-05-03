**MNIST Classification with Sequential CNN Model**
This project aims to classify handwritten digits from the MNIST dataset using a Sequential Convolutional Neural Network (CNN). The MNIST dataset is a widely used benchmark dataset in the field of machine learning and computer vision, consisting of 28x28 grayscale images of handwritten digits (0-9).

**Overview**
Utilized the MNIST dataset for training and testing.
Implemented a Sequential CNN model using TensorFlow/Keras to classify the digits.
Conducted inference on a single index of the test split to demonstrate model performance.
Dataset
The MNIST dataset consists of 60,000 training images and 10,000 testing images, each of size 28x28 pixels. It is widely used as a benchmark dataset for image classification tasks.

**Model Architecture**

The Sequential CNN model architecture used in this project is as follows:
Input Layer: Convolutional layer with ReLU activation.
Hidden Layers: Additional convolutional layers with ReLU activation, followed by max-pooling layers for downsampling.
Flatten Layer: Flattens the output of the convolutional layers to feed into the fully connected layers.
Fully Connected Layers: Dense layers with ReLU activation.
Output Layer: Dense layer with softmax activation for multi-class classification.

**Training:**
Run the notebook for training the model on the MNIST dataset.

**Inference:**
Use the trained model to make predictions on individual images from the test split.
**Results**
The model achieved an accuracy of 99.01% on the test split of the MNIST dataset.

# About Models

## First Model

### CNN Architecture

[CNNs](https://www.ibm.com/cloud/learn/convolutional-neural-networks) are a class of deep neural networks capable of recognizing and classifying particular features in images
and are primarily used for image analysis.

The term "[convolution](https://en.wikipedia.org/wiki/Convolution)" in CNN refers to the mathematical function of convolution, which is a special type of linear
operation in which two functions are multiplied to produce a third. The latter expresses how the form of one function
is modified by the other.

In other words, two images that can be represented as matrices are multiplied to produce one that will be used to
extract the features of the image in question.

There are two main parts in a CNN architecture:

- A convolution tool that separates and identifies the different features of the image for analysis in a process called
[Feature Extraction](https://en.wikipedia.org/wiki/Feature_extraction)
- A [fully connected layer](https://iq.opengenus.org/fully-connected-layer/) that uses the output of the convolution process and predicts the class of the image based on
the features extracted in the previous steps.

<p align="center">
    <img src="https://github.com/yassine-rd/bird_species_classification/blob/master/images/cnn.png" width="600" height="300"  alt="Implementation chart"/>
</p>

The CNN architecture proposed below gave significantly better results than the others.

It consists of four blocks :

- A first block with two convolutional layers followed by a max pooling
- A second one with two convolutional layers followed by a max pooling
- A third with two convolutional layers followed by max pooling and normalization
- The last block consists of a flattening layer, a 30% dropout and two activation layers with ReLU and softmax functions

### Classification model trained on 100 bird classes available [here](https://we.tl/t-QCHDSavrdz)

## VGG Model

We propose to establish a second model based on the VGG16 architecture

### Overview of the original VGG-16 model

The original model used in this analysis is a pre-trained VGG16 neural network. The latter is trained on an ImageNet
dataset of over 14 million images. For convenience, the size of the images used in VGG16 is the same as the images used
in this analysis: 224x224. VGG16 uses 5 blocks of 2D convolutional layers and 2D Max Pooling.

Here is a visual description of the VGG16 model:

<p align="center">
    <img src="https://github.com/yassine-rd/bird_species_classification/blob/master/images/vgg.png" width="600" height="300"  alt="Implementation chart"/>
</p>

### Improving the initial model

#### Augmentation

We made an augmentation with methods from the Keras library. The data is augmented in several ways, such as rotation,
zoom, and horizontal flipping. This allows the model to receive more images for training in order to have higher
accuracy without needing brand-new sources of data.

- Rotate the images randomly by 40 degrees
- Shift the image horizontally by 20%
- Shift the image vertically by 20%
- Zoom in on image by 20%
- Flip image horizontally

#### Changing the architecture

After freezing the original model layers, we added a pooling  and 3 activation layers. We also tested adding additional
Batch Normalization and 2D Convolution layers, but they had adverse effects on performance.

### VGG16 classification model trained on 100 bird classes available [here](https://we.tl/t-oe5yLGghEG)

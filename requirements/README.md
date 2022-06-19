# Requirements

Instructions given in this repository are meant to be used in a machine equiped with a
macOS operating system (in our case we used [macOS Monterey 12.4](https://support.apple.com/fr-fr/HT213257)). 
Readers should be aware that deep Learning is very computationally intensive and therefore 
should have access to a computer with the minimum requirements for their project.
You can consult [this](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/) guide in order to understand the requirements for using deep learning.

## Installing modules

Installing the modules needed to train a convolutional neural network (tensorflow and keras) can be troublesome as
it is required to properly install the graphic cards drives, CUDA, cuDNN and other dependencies. Special caution should be 
taken to install a tensorflow version that is compatible with the machine's CUDA and cuDNN versions, 
to install the right CUDA version for the machine's graphic card and to install tensorflow on the created virtual environment
as explained [here](https://github.com/yassine-rd/bird_species_classification/blob/master/requirements/TENSORFLOW.md).

All other modules are easier to install by typing on the terminal after activating the environment:

```javascript
pip install -r requirements.txt
```

## Python

Although all scripts used in this repository are of basic python code, if completely unfamiliarised with python 
a useful familiarization with python basic knowledge, can be of great use
(more specifically: [syntax]( https://www.w3schools.com/python/python_syntax.asp),
[for loops]( https://www.w3schools.com/python/python_for_loops.asp) and
[functions]( https://www.w3schools.com/python/python_functions.asp)).
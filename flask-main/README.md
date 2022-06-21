# Code

The code is divided as follows:

- The [app.py](https://github.com/yassine-rd/bird_species_classification/blob/master/flask-main/app.py) python file contains the necessary code to run the web application.
- The [process.py](https://github.com/yassine-rd/bird_species_classification/blob/master/flask-main/process.py) python file contains the necessary code for data pre-processing.
- The [cnn.py](https://github.com/yassine-rd/bird_species_classification/blob/master/flask-main/cnn.py) python file contains first model implementation.
- The [vgg.py](https://github.com/yassine-rd/bird_species_classification/blob/master/flask-main/vgg.py) python file contains vgg model implementation.
- The [utils.py](https://github.com/yassine-rd/bird_species_classification/blob/master/flask-main/utils.py) python file contains the necessary functions to plot various graphs.
- The [constants.py](https://github.com/yassine-rd/bird_species_classification/blob/master/flask-main/constants.py) python file contains the definition of various constants.

### Configuration
You can set up the configuration in the constants.py python file. In particular, you may need to modify:

- DIR_BASE : path to the folder where the dataset is.
- PATH_OUT : path to where first model will be saved.
- VGG_PATH_OUT : path to where vgg model will be saved.

### Steps to run the experiments
If you want to reproduce the experiments of the paper for a particular configuration:
1. Run the [cnn.py](https://github.com/yassine-rd/bird_species_classification/blob/master/flask-main/cnn.py) script to train base model.
```python
python cnn.py
```
2. Run the [cnn.py](https://github.com/yassine-rd/bird_species_classification/blob/master/flask-main/vgg.py) script to train vgg model.
```python
python vgg.py
```

You can directly run the web application without training new models
1. Run the [app.py](https://github.com/yassine-rd/bird_species_classification/blob/master/flask-main/app.py) script
```python
python app.py
```
2. Go to http://127.0.0.1:5000/ on your web browser
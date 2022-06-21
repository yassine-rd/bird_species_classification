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
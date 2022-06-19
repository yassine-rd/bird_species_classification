#Tensorflow installation instructions for arm64 architectures

##Step 1: Environment setup

* Download and install [Conda env](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh):
```javascript
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
```

* Create a new conda environment with python 3.9
```javascript
conda create -n web_app python=3.9
conda activate web_app
```

##Step 2: Install the TensorFlow dependencies

```javascript
conda install -c apple tensorflow-deps
```

##Step 3: Install base TensorFlow

```javascript
pip install tensorflow-macos
```

##Step 4: Install tensorflow-metal plugin

```javascript
pip install tensorflow-metal
```

##Verifying that tensorflow had been properly installed
* Open a python interpreter
```javascript
python
```

* Import tensorflow library
```javascript
import tensorflow as tf
```

* Check tensorflow version
```javascript
tf.__version__
```

* Check if it recognizes the gpu
```javascript
tf.config.list_physical_devices()
```
import os

# This will establish paths to our main directories
DIR_BASE = 'test_db'
DIR_TRAIN = os.path.join(DIR_BASE, 'train')
DIR_VALID = os.path.join(DIR_BASE, 'valid')
DIR_TEST = os.path.join(DIR_BASE, 'test')

# Image properties
IMAGE_SIZE = (224, 224)
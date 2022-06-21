import os

# Paths to our main directories
DIR_BASE = '../dataset'
DIR_TRAIN = os.path.join(DIR_BASE, 'train')
DIR_VALID = os.path.join(DIR_BASE, 'valid')
DIR_TEST = os.path.join(DIR_BASE, 'test')

# Paths to our models
PATH_OUT = '../models/base_model.h5'
VGG_PATH_OUT = '../models/vgg_model.h5'
LOCAL_MODEL = '../models/local_model_10.h5'

# Image properties
IMAGE_SIZE = (224, 224)
INPUT_IMAGE_SIZE = (224, 224, 3)

# Batches properties
BATCH_SIZE_32 = 32
BATCH_SIZE_64 = 64

# CNN properties
KERNEL_SIZE = (3, 3)
POOL_SIZE = (3, 3)

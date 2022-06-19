import os

# This will establish paths to our main directories
DIR_BASE = '../test_db_100'
DIR_TRAIN = os.path.join(DIR_BASE, 'train')
DIR_VALID = os.path.join(DIR_BASE, 'valid')
DIR_TEST = os.path.join(DIR_BASE, 'test')

# Image properties
IMAGE_SIZE = (224, 224)
INPUT_IMAGE_SIZE = (224, 224, 3)

# Batches properties
BATCH_SIZE_32 = 32
BATCH_SIZE_64 = 64

# CNN properties
KERNEL_SIZE = (3, 3)
POOL_SIZE = (3, 3)

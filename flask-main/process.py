from tensorflow.keras.preprocessing.image import ImageDataGenerator

from constants import *
from utils import *


# plt.style.use('fivethirtyeight')
# plot_classes_graph(DIR_TRAIN)
# plt.style.use('default')

df_train = process(DIR_TRAIN)
df_test = process(DIR_TEST)
df_valid = process(DIR_VALID)

# Returning the first five rows of our training data frame
df_train.head()

# Keras data generator can be used to pass the images through the convolutional neural network and apply
# rotation and zoom transformations to the images if desired.
# You can check https://keras.io/preprocessing/image/ for more transformations

# For the first cnn model
generator = ImageDataGenerator()
train_ds = generator.flow_from_dataframe(
    dataframe=df_train,
    x_col='filepaths',
    y_col='labels',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_64,
    subset='training',
    random_seed=42)

valid_ds = generator.flow_from_dataframe(
    dataframe=df_valid,
    x_col='filepaths',
    y_col='labels',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_32,
    subset='training')

test_ds = generator.flow_from_dataframe(
    dataframe=df_test,
    x_col='filepaths',
    y_col='labels',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_32)

# For the pre-trained vgg model
improved_gen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_gen = ImageDataGenerator(rescale=1. / 255)

improved_train_ds = improved_gen.flow_from_dataframe(
    dataframe=df_train,
    x_col='filepaths',
    y_col='labels',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_64,
    subset='training',
    random_seed=42)

improved_valid_ds = improved_gen.flow_from_dataframe(
    dataframe=df_valid,
    x_col='filepaths',
    y_col='labels',
    subset='training',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_32)

improved_test_ds = test_gen.flow_from_dataframe(
    dataframe=df_test,
    x_col='filepaths',
    y_col='labels',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE_32
)

# Defining bird classes
classes = list(train_ds.class_indices.keys())

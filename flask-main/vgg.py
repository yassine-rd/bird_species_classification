import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16

from constants import *
from process import improved_test_ds, improved_train_ds, improved_valid_ds

# Disabling tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Dataset classes count
nb_classes = len(improved_train_ds.class_indices.keys())

# Loading the pre-trained VGG16 from keras
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_IMAGE_SIZE)

# Freezing the extraction layers
vgg_model.trainable = False
for layer in vgg_model.layers:
    print(layer, layer.trainable)

# Adding a fully connected layer
layer0 = tf.keras.layers.Flatten()(vgg_model.output)
layer1 = tf.keras.layers.Dense(4096, activation='relu')(layer0)
layer2 = tf.keras.layers.Dense(4096, activation='relu')(layer1)
# Adding a dense layer with a value equal to the number of classes
out_layer = tf.keras.layers.Dense(nb_classes, activation='softmax')(layer2)
# Creating the model
vgg_model = tf.keras.Model(vgg_model.input, out_layer)
# Printing model summary
vgg_model.summary()

# Defining where to save the model after each epoch
filepath = "../models/vgg_model.h5"

# Adding a criteria to save only if there was an improvement in the model comparing
# to the previous epoch (in this case the model is saved if there was a decrease in the loss value)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Stopping training if there is no improvement in model for 5 consecutive epochs
early_stopping_monitor = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

callbacks_list = [checkpoint, early_stopping_monitor]

# Choosing the cost function to be optimized (Categorical Cross-Entropy)
cost_function = keras.losses.categorical_crossentropy

# Compiling the model with Adam optimizer and a learning rate of 0.0001
vgg_model.compile(loss=cost_function, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])

# Training the model
vgg_history = vgg_model.fit(
    improved_train_ds,
    epochs=50,
    verbose=1,
    validation_data=improved_valid_ds,
    callbacks=[callbacks_list])

# Evaluating the model on improved_test_ds
vgg_model.evaluate(improved_test_ds, use_multiprocessing=True, workers=10)

# Saving the trained model
vgg_model.save("../models/vgg_model.h5")

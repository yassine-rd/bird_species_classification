import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16

from constants import *
from process import improved_test_ds, improved_train_ds, improved_valid_ds

nb_classes = len(improved_train_ds.class_indices.keys())

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_IMAGE_SIZE)

vgg_model.trainable = False
for layer in vgg_model.layers:
    print(layer, layer.trainable)

layer0 = tf.keras.layers.Flatten()(vgg_model.output)
layer1 = tf.keras.layers.Dense(4096, activation='relu')(layer0)
layer2 = tf.keras.layers.Dense(4096, activation='relu')(layer1)
out_layer = tf.keras.layers.Dense(nb_classes, activation='softmax')(layer2)

vgg_model = tf.keras.Model(vgg_model.input, out_layer)
vgg_model.summary()

filepath = "../models/vgg_model.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping_monitor = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

callbacks_list = [checkpoint, early_stopping_monitor]
cost_function = keras.losses.categorical_crossentropy
vgg_model.compile(loss=cost_function, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])

vgg_history = vgg_model.fit(
    improved_train_ds,
    epochs=50,
    verbose=1,
    validation_data=improved_valid_ds,
    callbacks=[callbacks_list])

vgg_model.evaluate(improved_test_ds, use_multiprocessing=True, workers=10)
vgg_model.save("../models/vgg_model.h5")

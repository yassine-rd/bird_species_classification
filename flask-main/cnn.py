from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

from utils.constants import *
from process import train_ds, valid_ds, test_ds

nb_classes = len(train_ds.class_indices.keys())

# First model architecture
model = Sequential()

# Bloc 1
model.add(Conv2D(16, kernel_size=KERNEL_SIZE, padding='same', activation='relu', input_shape=INPUT_IMAGE_SIZE))
model.add(Conv2D(32, kernel_size=KERNEL_SIZE, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=POOL_SIZE))

# Bloc 2
model.add(Conv2D(32, kernel_size=KERNEL_SIZE, padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=KERNEL_SIZE, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=POOL_SIZE))

# Bloc 3
model.add(Conv2D(128, kernel_size=KERNEL_SIZE, padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=KERNEL_SIZE, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=POOL_SIZE))
model.add(BatchNormalization())

# Bloc 4
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(nb_classes, activation='softmax'))

model.summary()

filepath = '../models/cnn_mode.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping_monitor = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

callbacks_list = [checkpoint, early_stopping_monitor]

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

history = model.fit(train_ds, epochs=50, verbose=1, validation_data=valid_ds, callbacks=[callbacks_list])

model.evaluate(test_ds, use_multiprocessing=True, workers=10)

model.save('../models/cnn_model.h5')

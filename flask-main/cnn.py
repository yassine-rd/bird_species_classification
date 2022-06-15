import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from constants import *


def process(data):
    path = pathlib.Path(data)
    filepaths = list(path.glob(r"*/*.jpg"))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],
                      filepaths))
    df1 = pd.Series(filepaths, name='filepaths').astype(str)
    df2 = pd.Series(labels, name='labels')
    df = pd.concat([df1, df2], axis=1)
    return df


df_train = process(DIR_TRAIN)
df_test = process(DIR_TEST)
df_valid = process(DIR_VALID)

generator = ImageDataGenerator()
train_ds = generator.flow_from_dataframe(
    dataframe=df_train,
    x_col='filepaths',
    y_col='labels',
    target_size=(224, 224),
    batch_size=64,
    subset='training',
    random_seed=42)

valid_ds = generator.flow_from_dataframe(
    dataframe=df_valid,
    x_col='filepaths',
    y_col='labels',
    target_size=(224, 224),
    batch_size=32,
    subset='training')

test_ds = generator.flow_from_dataframe(
    dataframe=df_test,
    x_col='filepaths',
    y_col='labels',
    target_size=(224, 224),
    batch_size=32)

classes = list(train_ds.class_indices.keys())

model = Sequential()

# Bloc 1
model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

# Bloc 2
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

# Bloc 3
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(BatchNormalization())

# Bloc 4
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(10, activation='softmax'))

model.summary()

filepath = "/content/drive/MyDrive/test_db_100/my_model_100.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping_monitor = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

callbacks_list = [checkpoint, early_stopping_monitor]

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

history = model.fit(train_ds, epochs=30, verbose=1, validation_data=valid_ds, callbacks=[callbacks_list])

acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(len(acc))

# Plot: accuracy vs epoch
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

# Plot: loss values vs epoch
plt.plot(epochs, loss, 'r', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc=0)
plt.figure()
plt.show()

model.evaluate(test_ds, use_multiprocessing=True, workers=10)

model.save('models/local_model.h5')

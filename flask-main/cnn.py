from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

from constants import *
from utils import *
from process import train_ds, valid_ds, test_ds

# Disabling tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Dataset classes count
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
# Printing model summary
model.summary()

# Defining where to save the model after each epoch
filepath = PATH_OUT

# Adding a criteria to save only if there was an improvement in the model comparing
# to the previous epoch (in this case the model is saved if there was a decrease in the loss value)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Stopping training if there is no improvement in model for 5 consecutive epochs
early_stopping_monitor = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

callbacks_list = [checkpoint, early_stopping_monitor]

# Choosing the cost function to be optimized (Categorical Cross-Entropy)
cost_function = keras.losses.categorical_crossentropy

# Compiling the model with Adam optimizer and aa default learning rate (0.001)
model.compile(loss=cost_function, optimizer='adam', metrics=["accuracy"])

# Training the model
history = model.fit(
    train_ds,
    epochs=50,
    verbose=1,
    validation_data=valid_ds,
    callbacks=[callbacks_list])

# Evaluating the model on train_ds
model.evaluate(test_ds, use_multiprocessing=True, workers=10)

# Saving the trained model
model.save(PATH_OUT)

# Plotting accuracy and loss graphs
plt.style.use('fivethirtyeight')
plot_acc(history)
plot_loss(history)
plt.style.use('default')

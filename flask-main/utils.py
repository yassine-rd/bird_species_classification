import os
from matplotlib import pyplot as plt


# Defining a function that displays a graph representing the number of images per bird class in a directory
def calc_labels(directory):
    count, labels = {}, []
    for label in sorted(os.listdir(f'{directory}')):
        labels.append(label)
        count[label] = len(os.listdir(f'{directory}/{label}/'))
    return count, labels


def plot_classes_graph(directory):
    """
    Takes a directory as input and displays a graph representing the number of images per bird class in the directory.
    """
    labels_count = calc_labels(directory)[0]
    x, y = zip(*sorted(labels_count.items(), key=lambda e: e[0]))
    plt.figure(figsize=(25, 5))
    plt.title('Number of images per bird class')
    plt.xlabel('Bird class')
    plt.ylabel('Image count')
    plt.plot(x, y, linewidth=1.5, label='Number of images per bird class')
    plt.xticks(x, rotation=90)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.5)
    plt.show()


# Defining a function that plot accuracy variation on the train and validation set vs epochs count
def plot_acc(history):
    """
    Takes a model history as input and displays a graph representing accuracy variation on the train and validation
    set vs epochs count
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    # Plot: accuracy vs epoch
    plt.figure(figsize=(15, 3))
    plt.plot(epochs, acc, label='Training accuracy', linewidth=1.5)
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy', linewidth=1.5)
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.figure()
    plt.show()


# Defining a function that plot loss variation on the train and validation set vs epochs count
def plot_loss(history):
    """
    Takes a model history as input and displays a graph representing loss variation on the train and validation
    set vs epochs count
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    # Plot: loss values vs epoch
    plt.figure(figsize=(15, 3))
    plt.plot(epochs, loss, label='Training loss', linewidth=1.5)
    plt.plot(epochs, val_loss, 'r', label='Validation loss', linewidth=1.5)
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.figure()
    plt.show()

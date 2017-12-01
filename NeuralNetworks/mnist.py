import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_mnist_data():
    '''
    Loads the MNIST data and reformats it for use in all other functions.
    '''
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_dataset = mnist.train.images.reshape((55000,28*28))
    train_labels = mnist.train.labels
    valid_dataset = mnist.validation.images.reshape((5000,28*28))
    valid_labels = mnist.validation.labels
    test_dataset = mnist.test.images.reshape((10000,28*28))
    test_labels = mnist.test.labels
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
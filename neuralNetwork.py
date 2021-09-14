import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist # loading the dataset of images in pixels

(train_images,train_labels), (test_images, test_labels) = fashion_mnist.load_data() # split the data into testing and training
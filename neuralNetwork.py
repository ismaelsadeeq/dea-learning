import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
# import keras as keras

fashion_mnist = keras.datasets.fashion_mnist # loading the dataset of images in pixels

(train_images,train_labels), (test_images, test_labels) = fashion_mnist.load_data() # split the data into testing and training

print(train_images.shape)
print(type(train_images))

print(train_labels[:10])

class_names = ["T-shirt/top","Trouser","Pullover","Dress","coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255
test_images = test_images / 255

model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28,28)), # input layer (1)
  keras.layers.Dense(128,activation='relu'), # hidden layer (2)
  keras.layers.Dense(10,activation='softmax') # output layer (3)
])
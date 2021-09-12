from os import name
import numpy as np
import tensorflow as tf


name = tf.Variable("Abubakar Sadiq",tf.string) #scalar tensor

string = tf.Variable(["Abubakar Sadiq","tensor","222"],tf.string) #rank 1 tensor

string2 = tf.Variable([["Abubakar Sadiq",""],["test","vamos"]],tf.string) #rank 2 tensor

print(tf.rank(string2)) #tells you the rank of the sensor

number = tf.Variable(123,tf.int32)

print(tf.shape(string2))

#tensor shaping

tensor1 = tf.ones([1,2,3])

tensor2 = tf.reshape(tensor1,[2,3,1])

tensor3 = tf.reshape(tensor2,[3,-1])

print(tensor1)
print(tensor2)
print(tensor3)
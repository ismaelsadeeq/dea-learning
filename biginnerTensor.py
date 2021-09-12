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

#slicing tensors
matrix = [
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]
]

tensor4 = tf.Variable(matrix,dtype=tf.int32)
print(tf.rank(tensor4))
print(tensor4.shape)

#slicing
three = tensor4[0,2]
print("three is ",three)


row1 = tensor4[0]  # selects the first row
print(row1)

column1 = tensor4[:, 0]  # selects the first column
print(column1)

row_2_and_4 = tensor4[1::2]  # selects second and fourth row
print("second and fourth row ",row_2_and_4)

column_1_in_row_2_and_3 = tensor4[1:3,0]
print(column_1_in_row_2_and_3)

fives = tf.zeros([5,5,5,5])

# print(fives)

secondFives = tf.reshape(fives,[125,-1])

print(secondFives)


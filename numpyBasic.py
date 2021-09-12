
import math
import numpy as np

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    
    s = 1 /(1+math.exp(-x))
    
    return s

number = 1
print("sigmoid of "+str(number)+" is "+ str(basic_sigmoid(number)))

x = [1, 2, 3]
# basic_sigmoid(x) 
x = np.array([1, 2, 3])
print(np.exp(x))

# example of vector operation
y= np.array([1, 2, 3])
print (y + 3)

def vectorSigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+np.exp(-x))
    ### END CODE HERE ###
    return s

x = np.array([1, 2, 3])
print(vectorSigmoid(x))


def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    s = vectorSigmoid(x)
    ds = s*(1-s)
    ### END CODE HERE ###
    print("sigmoid is ",s)
    print("sigmoid gradiet is",ds)
    return ds

sigmoid_derivative(x)

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    ### END CODE HERE ###
    
    return v

# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
image = np.array(
                [
                  [  
                    [ 0.67826139,  0.29380381],
                    [ 0.90714982,  0.52835647],
                    [ 0.4215251 ,  0.45017551]
                  ],

                  [
                    [ 0.92814219,  0.96677647],
                    [ 0.85304703,  0.52351845],
                    [ 0.19981397,  0.27417313]
                  ],

                  [
                    [ 0.60659855,  0.00533165],
                    [ 0.10820313,  0.49978937],
                    [ 0.34144279,  0.94630077]
                  ]
                ]
)

print ("image2vector(image) = " + str(image2vector(image)))
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    
    # Divide x by its norm.
    x = x/x_norm
    ### END CODE HERE ###
    
    print(x.shape, x_norm.shape)

    return x

    
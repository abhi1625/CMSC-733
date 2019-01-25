"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True
# def conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling = True):
#
#     #Define SHape of Filter-Weights for Convolution
#     shape = [filter_size, filter_size, num_channels, num_filters]
#
#     #initialize Weights
#     weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
#     #initialize Bias
#     bias = tf.Variable(tf.constant(0.05, shape=[num_filters]))
#
#     #Define Conv_layer
#     layer = tf.nn.conv2d(input=img,
#                          filter=weights,
#                          strides=[1, 1, 1, 1],
#                          padding='SAME')
#
#     #Add Bias to each filter channel
#     layer += bias
#     #in case downsampling is True
#     if downsampling:
#         layer = tf.nn.max_pool(value = layer,
#                                ksize = [1,2,2,1],
#                                strides = [1,2,2,1],
#
#    #Rectfier : calculates max(x,0) for each input pixel
#
#     layer = tf.nn.relu(layer)
#
#     return layer, weights
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

#Define this layer to convert the optput of the convolution layers to the fully connected layers

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

#Define a fully connected layer for the neural Network
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer



def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    #Define Filter parameters for the first convolution layer
    filter_size1 = 5
    num_filters1 = 64

    #Define Filter parameters for the second convolution layer
    filter_size2 = 5
    num_filters2 = 64

    #Define number of neurons in hidden layer
    fc_size1 = 256
    fc_size2 = 128


    #Define number of class labels
    num_classes = 10
    #############################
    # Fill your network here!
    #############################

    #Reshape the image from the placeholder variables
    # x_image = tf.reshape(Img, [-1, ImageSize[0], ImageSize[1], ImageSize[2]])

    #Construct first convolution layer
    layer_conv1, weights_conv1 = \
        conv_layer(input=Img,
                   num_input_channels=ImageSize[2],
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
    #Construct second convolution layer
    layer_conv2, weights_conv2 = \
        conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling = True)
    #Define Flatenned Layer for communication between Convolution layers and Neural Net
    layer_flat, num_features = flatten_layer(layer_conv2)

    #Define the Neural Network's fully connected layers:
    #Hidden layers and output layers
    layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size1,
                         use_relu=True)
    layer_fc0 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size1,
                         num_outputs=fc_size2,
                         use_relu=True)

    layer_fc2 = new_fc_layer(input=layer_fc0,
                         num_inputs=fc_size2,
                         num_outputs=num_classes,
                         use_relu=False)

    #prLogits is defined as the final output of the neural network
    prLogits = layer_fc2
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(layer_fc2)

    return prLogits, prSoftMax

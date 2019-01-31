"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):

Abhinav Modi (abhi1625@umd.edu)
Graduate Student pursuing Masters in Robotics,
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
    # filter_size1 = 5
    # num_filters1 = 64

    #Define Filter parameters for the second convolution layer
    # filter_size2 = 5

    #Define number of neurons in hidden layer
    # fc_size1 = 256
    # fc_size2 = 128
    # num_filters2 = 64


    #Define number of class labels
    num_classes = 10
    #############################
    # Fill your network here!
    #############################

    #Reshape the image from the placeholder variables
    # x_image = tf.reshape(Img, [-1, ImageSize[0], ImageSize[1], ImageSize[2]])

    #Construct first convolution layer
    net = Img
    net = tf.layers.conv2d(inputs = net, name='layer_conv1', padding='same',filters = 32, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name ='layer_bn1')
    net = tf.nn.relu(net, name = 'layer_Relu1')
    # layer_conv1 = net
    net = tf.layers.conv2d(inputs = net, name = 'layer_conv2', padding= 'same', filters = 32, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn2')
    net = tf.nn.relu(net, name = 'layer_Relu2')
    # layer_conv2 = net
    net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)


    net = tf.layers.conv2d(inputs = net, name = 'layer_conv3', padding= 'same', filters = 64, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn3')
    net = tf.nn.relu(net, name = 'layer_Relu3')


    net = tf.layers.conv2d(inputs = net, name = 'layer_conv4', padding= 'same', filters = 32, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn4')
    net = tf.nn.relu(net, name = 'layer_Relu4')

    net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

    # layer_conv1, weights_conv1 = \
    #     conv_layer(input=Img,
    #                num_input_channels=ImageSize[2],
    #                filter_size=filter_size1,
    #                num_filters=num_filters1,
    #                use_pooling=True)
    # #Construct second convolution layer
    # layer_conv2, weights_conv2 = \
    #     conv_layer(input=layer_conv1,
    #                num_input_channels=num_filters1,
    #                filter_size=filter_size2,
    #                num_filters=num_filters2,
    #                use_pooling = True)
    # #Define Flatenned Layer for communication between Convolution layers and Neural Net
    # layer_flat, num_features = flatten_layer(layer_conv2)
    # net = tf.contrib.layers.flatten(net)
    net = tf.layers.flatten(net)

    #Define the Neural Network's fully connected layers:
    #                      num_inputs=num_features,
    #Hidden layers and output layers
    # layer_fc1 = new_fc_layer(input=layer_flat,
    #                      num_outputs=fc_size1,
    #                      use_relu=True)
    net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 128, activation = tf.nn.relu)

    net = tf.layers.dense(inputs = net, name ='layer_fc2',units=256, activation=tf.nn.relu)

    net = tf.layers.dense(inputs = net, name ='layer_fc3',units=128, activation=tf.nn.relu)

    net = tf.layers.dense(inputs = net, name='layer_fc_out', units = num_classes, activation = None)

    # layer_fc0 = new_fc_layer(input=layer_fc1,
    #                      num_inputs=fc_size1,
    #                      num_outputs=fc_size2,
    #                      use_relu=True)
    #
    # layer_fc2 = new_fc_layer(input=layer_fc0,
    #                      num_inputs=fc_size2,
    #                      num_outputs=num_classes,
    #                      use_relu=False)

    #prLogits is defined as the final output of the neural network
    # prLogits = layer_fc2
    prLogits = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax

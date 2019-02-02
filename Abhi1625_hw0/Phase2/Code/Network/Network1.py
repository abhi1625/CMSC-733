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

    net = tf.layers.flatten(net)

    #Define the Neural Network's fully connected layers:
    net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 128, activation = tf.nn.relu)

    net = tf.layers.dense(inputs = net, name ='layer_fc2',units=256, activation=tf.nn.relu)

    # net = tf.layers.dense(inputs = net, name ='layer_fc3',units=128, activation=tf.nn.relu)

    net = tf.layers.dense(inputs = net, name='layer_fc_out', units = num_classes, activation = None)

    #prLogits is defined as the final output of the neural network
    # prLogits = layer_fc2
    prLogits = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax

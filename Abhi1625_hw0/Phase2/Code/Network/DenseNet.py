"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2


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

def concatenation(nodes):
    return tf.concat(nodes,axis=3)

def denseBlock(Img, num_layers, len_dense,num_filters,kernel_size,downsampling = False):
    with tf.variable_scope("dense_unit"+str(num_layers)):
        nodes = []
        # img = tf.layers.conv2d(inputs = Img,padding = 'same', filters = num_filters, kernel_size = kernel_size, activation = None)
        img = tf.layers.conv2d(inputs = Img, padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)

        nodes.append(img)
        for z in range(len_dense):
            img = tf.nn.relu(Img)
            img = tf.layers.conv2d(inputs = img, padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
            net = tf.layers.conv2d(inputs = concatenation(nodes), padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
            nodes.append(net)
        return net
    # net = tf.layers.conv2d(inputs = net, name=name+'split_0', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    # net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = name+'split_n')

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    #Define Filter parameters for the first convolution layer block
    filter_size1 = 5
    num_filters1 = 16
    n1 = 1                  #number of residual blocks in each convolution layer blocks

    #Define Filter parameters for the second convolution layer block
    filter_size2 = 5
    num_filters2 = 32
    n2 = 2                 #number of residual blocks in each convolution layer blocks

    #Define Filter parameters for the second convolution layer block
    filter_size = 3
    num_filters = 16
    n = 1                  #number of residual blocks in each convolution layer blocks


    #Define number of class labels
    num_classes = 10
    #############################
    # Fill your network here!
    #############################

    #Define net placeholder
    net = Img

    net = tf.layers.conv2d(net,num_filters1,kernel_size = filter_size1,activation = None)

    #Construct first convolution block of n residual blocks
    net  = denseBlock(Img = net, num_layers = 1, len_dense = 5,num_filters = num_filters1,kernel_size =filter_size1 ,downsampling = False)
    # net = n_res_block(net, num_filters = num_filters4, kernel_size = filter_size4, n_blocks = n4, b=4, downsampling =True)

    #Define flatten_layer
    net = tf.layers.flatten(net)

    #Define the Neural Network's fully connected layers:
    net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 128, activation = tf.nn.relu)

    # net = tf.layers.dense(inputs = net, name ='layer_fc2',units=128, activation=tf.nn.relu)

    # net = tf.layers.dense(inputs = net, name ='layer_fc3',units=64, activation=tf.nn.relu)

    net = tf.layers.dense(inputs = net, name='layer_fc_out', units = num_classes, activation = None)


    #prLogits is defined as the final output of the neural network
    # prLogits = layer_fc2
    prLogits = net
    #prSoftMax is defined as normalized probabilities of the output of the neural network
    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax

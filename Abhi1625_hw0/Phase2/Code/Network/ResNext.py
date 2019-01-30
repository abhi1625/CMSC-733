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

#Define Residual Network Block
def name_block(s,b):
    """
    s is the integer for the number of each set of residual blocks
    """
    s = 'set'+str(b)+'block'+str(s)
    return s

def concatenation(out):
    return tf.concat(out, axis = 3)

def ideal_block(Img, num_filters1, num_filters2, kernel_size1, kernel_size2,s,b):
    net = Img
    name = name_block(s=s,b=b)
    # net = tf.layers.conv2d(inputs = net, name=name+'layer_res_conv_0', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    # net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = name+'layer_bn0')
    # I_store = net

    net = tf.layers.conv2d(inputs = net, name=name +'conv_1', padding='same',filters = num_filters1, kernel_size = kernel_size1, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name =name +'layer_bn1')
    net = tf.nn.relu(net, name = name +'layer_Relu1')

    #Define 2nd Layer of the convolution
    net = tf.layers.conv2d(inputs = net, name=name+'conv_2', padding='same',filters = num_filters2, kernel_size = kernel_size2, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name =name +'layer_bn2')
    return net

def split(Img, cardinality, num_filters1, num_filters2, kernel_size1, kernel_size2, b):
    """
    Img is the input from each path taken by the input
    cardinality is the number of paths to be taken
    """
    # out = ideal_block(Img = Img, num_filters1 = num_filters1, num_filters2 = num_filters2, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, s = 0, b=b)
    out = list()
    for i in range(cardinality):
        # net = Img
        net = ideal_block(Img = Img, num_filters1 = num_filters1, num_filters2 = num_filters2, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, s = i, b=b)
        out.append(net)

    return concatenation(out)

def merge_block(Img, cardinality,num_filters, num_filters1, num_filters2,kernel_size, kernel_size1, kernel_size2, b):
    net = Img
    name = name_block(s=b,b=b)
    net = tf.layers.conv2d(inputs = net, name=name+'merge_0', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = name+'merge_bn0')
    I_store = net

    net = split(Img = net, cardinality = cardinality, num_filters1 = num_filters1, num_filters2 = num_filters2, kernel_size1 = kernel_size1, kernel_size2 = kernel_size2, b = b)
    net = tf.layers.conv2d(inputs = net, name=name+'split_0', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = name+'split_n')

    out = tf.math.add(net, I_store)

    net = tf.nn.relu(out, name = name +'layer_Relu2')

    return net



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
    filter_size1 = 1
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
    #Construct first convolution block of n residual blocks
    net = merge_block(Img = net, cardinality = 5,num_filters = num_filters, num_filters1 = num_filters1, num_filters2 = num_filters2,kernel_size = filter_size, kernel_size1= filter_size1, kernel_size2= filter_size2, b = 1)
    net = merge_block(Img = net, cardinality = 3,num_filters = num_filters, num_filters1 = num_filters1, num_filters2 = 16,kernel_size = filter_size, kernel_size1= filter_size1, kernel_size2= filter_size2, b = 2)

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

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

def res_block(Img, num_filters, kernel_size,s,b, downsampling = True):
    """
    net is a MiniBatch of the current image
    num_filters is number of filters used in this layer
    kernel_size is the size of filter used
    s is the integer number of the block
    b is the integer number of the set of blocks with same parameters
    """
    name = name_block(s=s,b=b)
    im_store = tf.layers.conv2d(inputs = Img, name=name+'layer_res_conv_0', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    im_store = tf.layers.batch_normalization(inputs = im_store,axis = -1, center = True, scale = True, name = name+'layer_bn0')
    I_store = im_store

    #Define 1st layer of convolution
    net = tf.layers.conv2d(inputs = Img, name=name +'layer_res_conv_1', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name =name +'layer_bn1')
    net = tf.nn.relu(net, name = name +'layer_Relu1')

    #Define 2nd Layer of the convolution
    net = tf.layers.conv2d(inputs = net, name=name+'layer_res_conv_2', padding='same',filters = num_filters, kernel_size = kernel_size, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name =name +'layer_bn2')

    if downsampling:
        net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)
        I_store = tf.layers.max_pooling2d(inputs = I_store, pool_size = 2, strides = 2)
        print('\n')
        print('max_pool is on')
    out = tf.math.add(net, I_store)

    net = tf.nn.relu(out, name = name +'layer_Relu2')

    return net

def n_res_block(net, num_filters, kernel_size, n_blocks, b, downsampling =False):
    """
    net is a MiniBatch of the current image
    num_filters is number of filters used in this layer
    kernel_size is the size of filter used
    n_blocks is number of residual blocks with same num_filters and kernel_size
    b is the integer number of the set of blocks being used
    """

    for i in range(n_blocks):
        net = res_block(Img = net, num_filters = num_filters, kernel_size = kernel_size,s = i,b = b, downsampling = downsampling )
        net = net
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
    filter_size1 = 5
    num_filters1 = 16
    n1 = 1                  #number of residual blocks in each convolution layer blocks

    #Define Filter parameters for the second convolution layer block
    filter_size2 = 5
    num_filters2 = 64
    n2 = 2                 #number of residual blocks in each convolution layer blocks

    #Define Filter parameters for the second convolution layer block
    filter_size3 = 4
    num_filters3 = 16
    n3 = 1                  #number of residual blocks in each convolution layer blocks

    # filter_size4 = 3
    # num_filters4 = 16
    # n4 = 1                  #number of re
    #Define number of neurons in hidden layer
    # fc_size1 = 256
    # fc_size2 = 128


    #Define number of class labels
    num_classes = 10
    #############################
    # Fill your network here!
    #############################
    #Define net placeholder
    net = Img
    #Construct first convolution block of n residual blocks
    net = n_res_block(net, num_filters = num_filters1, kernel_size = filter_size1, n_blocks = n1, b=1, downsampling =False)
    net = n_res_block(net, num_filters = num_filters2, kernel_size = filter_size2, n_blocks = n2, b=2, downsampling =False)
    net = n_res_block(net, num_filters = num_filters3, kernel_size = filter_size3, n_blocks = n3, b=3, downsampling =False)
    # net = n_res_block(net, num_filters = num_filters4, kernel_size = filter_size4, n_blocks = n4, b=4, downsampling =True)

    #Define flatten_layer
    net = tf.layers.flatten(net)

    #Define the Neural Network's fully connected layers:
    #                      num_inputs=num_features,
    #Hidden layers and output layers
    # layer_fc1 = new_fc_layer(input=layer_flat,
    #                      num_outputs=fc_size1,
    #                      use_relu=True)
    # net = tf.layers.dense(inputs = net, name ='layer_fc1', units = 256, activation = tf.nn.relu)
    #
    # net = tf.layers.dense(inputs = net, name ='layer_fc2',units=128, activation=tf.nn.relu)

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

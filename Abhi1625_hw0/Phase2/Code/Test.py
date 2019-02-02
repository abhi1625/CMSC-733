#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 21:16:41 2019

@author: kartikmadhira
"""

#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
import time
import tensorflow as tf
import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.DenseNet import CIFAR10Model
from Misc.MiscUtils import *
from PIL import Image
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util




# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs:
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.png'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')

    return ImageSize, DataPath

def ReadImages(ImageSize, DataPath):
    """
    Inputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """

    ImageName = DataPath

    I1 = cv2.imread(ImageName)

    if(I1 is None):
        # OpenCV returns empty list if image is not read!
        print('ERROR: Image I1 cannot be read')
        sys.exit()


    I1 = (I1-np.mean(I1))/255
    I1Combined = np.expand_dims(I1, axis=0)

    return I1Combined, I1


def TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    ImageSize is the size of the imge
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    # Predict output with forward pass, MiniBatchSize for Test is 1
    _, prSoftMaxS = CIFAR10Model(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()


    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        OutSaveT = open(LabelsPathPred, 'w')

        for count in tqdm(range(np.size(DataPath))):
            DataPathNow = DataPath[count]
            Img, ImgOrg = ReadImages(ImageSize, DataPathNow)
            FeedDict = {ImgPH: Img}
            PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))

            OutSaveT.write(str(PredT)+'\n')

        OutSaveT.close()

def Accuracy(Pred, GT):
    """
    Inputs:
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelTest, LabelPred

# def ConfusionMatrix(LabelsTrue, LabelPred):
#     """
#     LabelsTrue - True labels
#     LabelsPred - Predicted labels
#     """
#
#     # Get the confusion matrix using sklearn.
#     cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
#                           y_pred=LabelPred)  # Predicted class.
#
#     # Print the confusion matrix as text.
#     for i in range(10):
#         print(str(cm[i, :]) + ' ({0})'.format(i))
#
#     # Print the class-numbers for easy reference.
#     class_numbers = [" ({0})".format(i) for i in range(10)]
#     print("".join(class_numbers))
#
#     print('Accuracy: '+ str(Accuracy(LabelPred, LabelsTrue)), '%')
def ConfusionMatrix(LabelsTrue, LabelsPred, num_classes):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """
    print('length = '+ str(len(LabelsPred)))
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: ' + str(Accuracy(LabelsPred, LabelsTrue)), '%')

    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """
    accTestOverEpochs=np.array([0,0])

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumEpochs', dest='NumEpochs',type=int, default=5, help='Path to load images from, Default:BasePath')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/abhinav/CMSC-733/Abhi1625_hw0/Phase2/Checkpoints/', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/abhinav/CMSC-733/Abhi1625_hw0/Phase2/CIFAR10/Test/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    start = time.time()
    for epoch in range(NumEpochs):
        # Parse Command Line arguments
        tf.reset_default_graph()

        ModelPath = '/home/abhinav/CMSC-733/Abhi1625_hw0/Phase2/Checkpoints/'+str(epoch)+'model.ckpt'
        print(ModelPath)
        BasePath = Args.BasePath
        LabelsPath = Args.LabelsPath

        # Setup all needed parameters including file reading
        ImageSize, DataPath = SetupAll(BasePath)

        # Define PlaceHolder variables for Input and Predicted output
        ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
        LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

        TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred)

        # Plot Confusion Matrix
        LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
        accuracy=Accuracy(LabelsTrue, LabelsPred)
        accTestOverEpochs=np.vstack((accTestOverEpochs,[epoch,accuracy]))
        plt.xlim(0,60)
        plt.ylim(0,100)
        plt.xlabel('Epoch')
        plt.ylabel('Test accuracy')
        plt.subplots_adjust(hspace=0.6,wspace=0.3)
        plt.plot(accTestOverEpochs[:,0],accTestOverEpochs[:,1])
        plt.savefig('Graphs/test/Epochs'+str(epoch)+'.png')
        plt.close()
    end = time.time()
    infer_time = (end-start)/10000
    print('inference time:',infer_time)
    ConfusionMatrix(LabelsTrue, LabelsPred,10)

if __name__ == '__main__':
    main()

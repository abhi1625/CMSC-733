#import numpy as np
#import cv2
#import random
#import skimage
#import PIL
import sys

sys.dont_write_bytecode = True

def StandardizeInputs(img):
	img /= 255
	img -= 0.5
	img *= 2
	

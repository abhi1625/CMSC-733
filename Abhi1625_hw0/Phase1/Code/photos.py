import cv2
import numpy as np

img1 = cv2.imread("/home/abhinav/CMSC-733/Abhi1625_hw0/Phase1/BSDS500/Images/10.jpg")
img2 = cv2.imread("/home/abhinav/CMSC-733/Abhi1625_hw0/Phase1/Code/10/PbLite_10canny=0.3.png")

cv2.imwrite("web.png",np.hstack([img1,img2]))

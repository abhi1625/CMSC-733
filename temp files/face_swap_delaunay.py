#!/usr/bin/env python
# coding: utf-8

# In[85]:


# import tensorflow as tf
import cv2
import sys
import os
# import glob
# import Misc.ImageUtils as iu
# import random
import matplotlib.pyplot as plt
import numpy as np
# import time
# import argparse
# from StringIO import StringIO
# import string
# import math as m
# from tqdm import tqdm
import dlib
from imutils import face_utils


# In[86]:


# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color, 0 )


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList();
    #print(triangleList)

    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :

            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)



# In[87]:


# Draw delaunay triangles
def getTriMatches(r,subdiv,indexDict,firstPoints,secondPoints) :
    pointsArray=[(0,0),(0,0),(0,0)]

    firstTriangles=[]
    secTriangles=[]
    triangleList = subdiv.getTriangleList();
   # print(triangleList)
    for t in triangleList :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])




        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            first=indexDict[pt1]
            second=indexDict[pt2]
            third=indexDict[pt3]

            ptI11 = (firstPoints[first][0], firstPoints[first][1])
            ptI12 = (firstPoints[second][0], firstPoints[second][1])
            ptI13 = (firstPoints[third][0], firstPoints[third][1])

            ptI21 = (secondPoints[first][0], secondPoints[first][1])
            ptI22 = (secondPoints[second][0], secondPoints[second][1])
            ptI23 = (secondPoints[third][0], secondPoints[third][1])
                #print('lol1')
            tempFirst=[ptI11,ptI12,ptI13]
            tempSecond=[ptI21,ptI22,ptI23]


#             firstTriangles=np.vstack((firstTriangles,tempFirst))
            firstTriangles.append(tempFirst)

#             print(firstTriangles)
            secTriangles.append(tempSecond)

#             secTriangles=np.vstack((secTriangles,tempSecond))
    #print(firstTriangles)
    #np.delete(firstTriangles,0)
    #np.delete(secTriangles,0)

    return np.asarray(firstTriangles),np.asarray(secTriangles)


# In[88]:



def getPoints(img,detector,predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if np.shape(rects)[0] == 0:
        shape = 0
        feature_found = False
    else:
        feature_found = True
#     print('rects')
#     print(rects)
        points=0
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(img, rect)
            shape = face_utils.shape_to_np(shape)
    return shape, feature_found





def get_face_params(img,feature_points, allowance=10):
#   input---
#   img: input image
#   feature_points: numpy array of feature points from dlib predictor
#   allowance: extra region to house face in a rectangle

#   output---
#   cropped face
#   shifted feature points w.r.t cropped face
#   height and width of cropped image

    im_w, im_h = img.shape[:2]
    # finding leftmost point in the image
#     print(feature_points)
    left, top = np.min(feature_points.astype(np.uint8), 0)
    if left<top:
        top = left
    else:
        left = top
    # finding bottom right point in the image
    right, bottom = np.max(feature_points.astype(np.uint8), 0)
    if right>bottom:
        bottom = right
    else:
        right = bottom

    x, y = max(0, left-allowance), max(0, top-allowance)
    w, h = min(right+allowance, im_h)-x, min(bottom+allowance, im_w)-y
    new_points = feature_points - np.asarray([[x, y]])
    rect = (x, y, w, h)
    cropped_img = img[y:y+h, x:x+w]
    return new_points, rect, cropped_img



# Draw delaunay triangles
def drawSingle(img, pts, delaunay_color ) :

    #print(triangleList)

    size = img.shape
    r = (0, 0, size[1], size[0])



    pt1 = tuple(pts[9][0])
    pt2 = tuple(pts[9][1])
    pt3 = tuple(pts[9][2])

    if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :

        cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
    plt.imshow(img)




def calcBaryCentres(dstImg,srcImg,firstTriangle,secondTriangle):
#     secondImage=dstImg.copy()
    rows, cols = np.shape(dstImg)[:2]
    # print(rows)
    # print(cols)
    secondImage = np.zeros((rows, cols, 3), dtype=int)
    firstCopy=srcImg.copy()
    shape=dstImg.shape
    xmin=0
    ymin=0
    xmax=shape[0]
    ymax=shape[1]
    X, Y = np.mgrid[xmin:xmax, ymin:ymax]
    locations=np.vstack((X.ravel(), Y.ravel()))
    locations=np.vstack((locations,np.ones(shape=(1,ymax*ymax))))
    prevCentres=0
    for i,_ in enumerate(firstTriangle):
        print(i)
        #getting BDelta function
        bDelta=[[secondTriangle[i][0][0],secondTriangle[i][0][1],1],
        [secondTriangle[i][1][0],secondTriangle[i][1][1],1],
       [secondTriangle[i][2][0],secondTriangle[i][2][1],1]]
        #transposing to get the proper format
        bDelta=np.matrix(bDelta).transpose()
#         print(bDelta)
        #calculate barycentres and then filter
        try:
            inverse = np.linalg.inv(bDelta)
        except np.linalg.LinAlgError:
            # Not invertible. Skip this one.
            pass
        else:
    # continue with what you were doing


            baryCentres=np.matmul(np.linalg.inv(bDelta),locations)

            #threshold
            check=np.where((baryCentres[0]>=0)&(baryCentres[0]<=1)
                           &(baryCentres[1]>=0)&(baryCentres[1]<=1)
                           &(baryCentres[2]>=0)&(baryCentres[2]<=1))
            #access correspoinding pixels that follow the barycentric constraints
            pts_dest = locations[:2,check[1]]

            aDelta=[[firstTriangle[i][0][0],firstTriangle[i][0][1],1],
            [firstTriangle[i][1][0],firstTriangle[i][1][1],1],
           [firstTriangle[i][2][0],firstTriangle[i][2][1],1]]

            #transposing to get the proper format
            aDelta=np.matrix(aDelta).transpose()

            threshBaryCentres=baryCentres[:,check[1]]

            #calculate barycentres and then filter
            newCentres=np.matmul(aDelta,threshBaryCentres)

            #newCentres=np.divide(newCentres,newCentres[2])
            #print(firstCopy.shape[0],firstCopy.shape[1])
            # print(np.shape(secondImage))
            # print('space')
            # print(np.shape(secondImage))
            #print(newCentres[0].astype(int))
                #,newCentres[0].astype(int))
            # if(firstCopy.shape[0]==200 and firstCopy.shape[1]==200):
            # if (i==0):
            #     prevCentres=newCentres
            # if(newCentres.size>0):
            #     print(newCentres.size,np.max(newCentres[0].astype(int)),np.max(newCentres[1].astype(int)))
            #     if((firstCopy.shape[0]<(np.max(newCentres[0].astype(int))))
            #      or (firstCopy.shape[0]<(np.max(newCentres[1].astype(int))))):

            firstImageValues=firstCopy[newCentres[1].astype(int),newCentres[0].astype(int)]
            secondImage[pts_dest[1].astype(int),pts_dest[0].astype(int)]=firstImageValues
            prevCentres=newCentres
        # else:
            firstImageValues=firstCopy[prevCentres[1].astype(int),prevCentres[0].astype(int)]
            secondImage[pts_dest[1].astype(int),pts_dest[0].astype(int)]=firstImageValues


                #print(prevCentres)


           # if(np.shape(secondImage)[0]<=pts_dest[0] and np.shape(secondImage)[1]<pts_dest[0]):
            # secondImage[pts_dest[1].astype(int)-1,pts_dest[0].astype(int)-1]=firstImageValues

    return secondImage






def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel,iterations=1)

    return mask


# In[110]:


## Copy-and-paste
def apply_mask(img, mask):
    """ Apply mask to supplied image
    :param img: max 3 channel image
    :param mask: [0-255] values in mask
    :returns: new image with mask applied
    """
    masked_img=cv2.bitwise_and(img,img,mask=mask)

    return masked_img


# In[111]:


def correct_colours(im1, im2, landmarks1):
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount -= 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
#     im2_blur = im2

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur.astype(int)
    im2_blur += 128*(im2_blur <= 1)

    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def swap_faces(firstImage,secondImage,detector,predictor, resize_val):
    # takes the face in firstImage and replaces the face in second image

    # ind = 0
    # resize_val = 200
    firstImage = cv2.resize(firstImage,(resize_val,resize_val))
    secondImage = cv2.resize(secondImage,(resize_val,resize_val))

    #get the sizes of the images
    firstSize=firstImage.shape
    firstRect= (0, 0, firstSize[1], firstSize[0])

    secondSize=secondImage.shape
    secondRect= (0, 0, secondSize[1], secondSize[0])
    canvasRect=(0,0,firstSize[1]+secondSize[1],firstSize[0]+secondSize[0])


    #create instances of subdiv
    firstSubdiv = cv2.Subdiv2D(firstRect);
    secondSubdiv = cv2.Subdiv2D(secondRect);
    canvasSubdiv=cv2.Subdiv2D(canvasRect);


    #get the facial markers from the images.
    firstArray, flag1=getPoints(firstImage,detector,predictor)
    secondArray, flag2=getPoints(secondImage,detector,predictor)

    if(flag1==False or flag2==False):
        # print('skipping frame, 68 features not found')
        features_found = False
        output = 0
        return output,features_found
    else:
        features_found = True
    firstArray, src_shape, src_face = get_face_params(firstImage.copy(), firstArray)
    secondArray, dst_shape, dst_face = get_face_params(secondImage.copy(), secondArray)

    firstCopy=src_face.copy()
    secondCopy=dst_face.copy()

    #get the average of the points
    averagePoints=(firstArray+secondArray)/2

    points=[]
    for i in range(0,68):
        points.append((averagePoints[i][0],averagePoints[i][1],))

    indexDict={}
    i=0
    for p in points:
        indexDict[p]=i
        i+=1

    for p in points :
        #print(p)
        canvasSubdiv.insert(p)
    firstTriangle,secondTriangle=getTriMatches(canvasRect,canvasSubdiv,indexDict,firstArray,secondArray)

    dstImage=secondCopy.copy()



    shape=dstImage.shape
    xmin=0
    ymin=0
    xmax=shape[0]
    ymax=shape[1]

    X, Y = np.mgrid[xmin:xmax, ymin:ymax]
    locations=np.vstack((X.ravel(), Y.ravel()))
    # print(locations)
    locations=np.vstack((locations,np.ones(shape=(1,ymax*ymax))))

    warped_src_face = calcBaryCentres(secondCopy.copy(),firstCopy.copy(),firstTriangle,secondTriangle)

    # plt.rcParams["figure.figsize"] = (5,5)

    w, h = dst_face.shape[:2]
    ###########################################################################
    #################### Mask for blending ####################################
    ###########################################################################

    ####### Mask for blending
    mask = mask_from_points((w, h), secondArray)
    # plt.imshow(mask)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask*mask_src, dtype=np.uint8)

    warped_src_face = apply_mask(warped_src_face, mask.copy())
    dst_face_masked = apply_mask(dst_face, mask.copy())
    warped_src_face = warped_src_face.astype(np.uint8)
    # np.asarray(warped_src_face,dtype=np.uint8)
    print(warped_src_face.dtype)
    warped_src_face = correct_colours(dst_face_masked, warped_src_face, secondArray)

    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

    x, y, w, h = dst_shape
    dst_img_cp = secondImage.copy()
    dst_img_cp[y:y+h, x:x+w] = output
    output = dst_img_cp

    return output, features_found



def main():
    print('FaceSwap initiated')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor.dat')
    # img_tar = cv2.imread('./Data/kartik.jpeg')
    img_src = cv2.imread('./TestSet_P2/Rambo.jpg')
    # delaunay_color = (255,255,255)

    # points_color = (0, 0, 255)
    ind = 1
    # cap1 = cv2.VideoCapture('./TestSet_P2/Test1.mp4')
    cap2 = cv2.VideoCapture('./TestSet_P2/Test1.mp4')

    while(True):
        # ret1,img_src = cap1.read()
        ret2,img_tar = cap2.read()
        if((not ret2)):# or (not ret2)):
            break

        shape1= np.shape(img_src)[:2]
        shape2= np.shape(img_tar)[:2]
        a = np.minimum(shape1,shape2)
        a = np.minimum(a[0],a[1])


        output, flag = swap_faces(img_src, img_tar, detector,predictor,a)


        if flag == False:
            print('skipping frame, 68 feature not found')
            cv2.imwrite('./delaunay_result1/'+str(ind)+'.jpg',img_tar)
            continue

        cv2.imwrite('./delaunay_result1/'+str(ind)+'.jpg',output)
        ind+=1
        print(ind)

if __name__ == '__main__':
    main()

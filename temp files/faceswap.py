#!/usr/bin/env python
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils
import math
import sys
import matplotlib.pyplot as plt

def features(img,detector,predictor):
    #initialize facial detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(200,200))
    img = cv2.resize(img,(200,200))

    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor('/home/abhinav/CMSC-733/Abhi1625_p2/shape_predictor.dat')
    rects = detector(gray,1)


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for (i,rect) in enumerate(rects):

        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)


        (x,y,w,h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


        for (x,y) in shape:
            cv2.circle(img,(x,y),2,(0,0,255),-1)
    return img, shape


#Thin Plate Splines
def U(r):
    return (r**2)*(math.log(r**2))
def TPS_generate(source,target):
    P = np.append(source,np.ones([source.shape[0],1]),axis=1)
    P_Trans = P.T
    Z = np.zeros([3,3])
    K = np.zeros([source.shape[0],source.shape[0]])
    for p in range(source.shape[0]):
        K[p] = [U(np.linalg.norm((-source[p]+source[i]),ord =2)+sys.float_info.epsilon) for i in range(source.shape[0])]

    M = np.vstack([np.hstack([K,P]),np.hstack([P_Trans,Z])])
    lam = 500
    I = np.identity(M.shape[0])
    L = M+lam*I
    L_inv = np.linalg.inv(L)
    V = np.concatenate([np.array(target),np.zeros([3,])])
    V.resize(V.shape[0],1)
    weights = np.matmul(L_inv,V)
    return weights,K


def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel,iterations=1)

    return mask

def fxy(pt1,pts2,weights):
    K = np.zeros([pts2.shape[0],1])
    for i in range(pts2.shape[0]):
        K[i] = U(np.linalg.norm((pts2[i]-pt1),ord =2)+sys.float_info.epsilon)
    f = weights[-1] + weights[-3]*pt1[0] +weights[-2]*pt1[1]+np.matmul(K.T,weights[0:-3])
    return f


def warp_tps(img_source,img_target,points1,points2,weights_x,weights_y,mask):
    xy1_min = np.float32([min(points1[:,0]),min(points1[:,1])])
    xy1_max = np.float32([max(points1[:,0]),max(points1[:,1])])

    x = np.arange(xy1_min[0],xy1_max[0]).astype(int)
    y = np.arange(xy1_min[1],xy1_max[1]).astype(int)

    X,Y = np.mgrid[x[0]:x[-1]+1,y[0]:y[-1]+1]

    # X,Y = np.mgrid[0:src_shape[2],0:src_shape[3]]
    pts_src = np.vstack((X.ravel(),Y.ravel()))
    xy = pts_src.T
    u = np.zeros_like(xy[:,0])
    v = np.zeros_like(xy[:,0])
    # print(u.shape)
    # print(v.shape)
    for i in range(xy.shape[0]):
        u[i] = fxy(xy[i,:],points1,weights_x)
    # u[u<xy2_min[0]]=xy2_min[0]
    # u[u>xy2_max[0]]=xy2_max[0]
    for j in range(xy.shape[0]):
        v[j] = fxy(xy[j,:],points1,weights_y)
    # v[v<xy2_min[1]]=xy2_min[1]
    # v[v>xy2_max[1]]=xy2_max[1]
#     print(u.shape)
#     print(img_source.shape)
    warped_img = img_source.copy()
    mask_warped_img = np.zeros_like(warped_img[:,:,0])
    for a in range(u.shape[0]):
    #     for b in range(v.shape[0]):
    #     warped_img[xy[a,1],xy[a,0],:] = warped_src_face[v[a],u[a],:]
        if mask[v[a],u[a]]>0:
            warped_img[xy[a,1],xy[a,0],:] = img_target[v[a],u[a],:]
            mask_warped_img[xy[a,1],xy[a,0]] = 255

    return warped_img, mask_warped_img


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

def swap(img_source,img_target,points1,points2):
    weights_x,K = TPS_generate(points1,points2[:,0])
    weights_y,K = TPS_generate(points1,points2[:,1])
    # plt.imshow(K)

    w, h = img_target.shape[:2]
    # ## Mask for blending
    mask = mask_from_points((w, h), points2)
    # plt.imshow(mask)
    # mask.shape

    warped_img, mask_warped_img = warp_tps(img_source,img_target,points1,points2,weights_x,weights_y,mask)
    # plt.imshow(warped_img)
    # plt.imshow(mask_warped_img)
    # mask_warped_img.shape


    ##Poisson Blending
    r = cv2.boundingRect(mask_warped_img)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_img.copy(), img_source, mask_warped_img, center, cv2.NORMAL_CLONE)
    return output

def main():
    #Define Face Detector and Predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/home/abhinav/CMSC-733/Abhi1625_p2/shape_predictor.dat')

    #Setup input video
    cap = cv2.VideoCapture('/home/abhinav/CMSC-733/Abhi1625_p2/john.mp4')
    i=0
    while(True):
        ret,img_src= cap.read()
        # plt.imshow(img_src)
        # plt.show()
        print(img_src.dtype)
        # img_src = cv2.imread('/home/abhinav/CMSC-733/Abhi1625_p2/chahat_c.jpeg')
        img_tar = cv2.imread('/home/abhinav/CMSC-733/Abhi1625_p2/imgs/joker.jpg')
        img_source = cv2.resize(img_src,(200,200))
        # img_source = cv2.cvtColor(img_source,cv2.COLOR_BGR2RGB)
        img_target = cv2.resize(img_tar,(200,200))
        # img_target = cv2.cvtColor(img_target,cv2.COLOR_BGR2RGB)
        img1 = img_src.copy()
        img2 = img_tar.copy()

        img1,points1 = features(img1,detector,predictor)
        img2,points2 = features(img2,detector, predictor)

        output1 = swap(img_source.copy(),img_target.copy(),points1,points2)
        output2 = swap(img_target.copy(),img_source.copy(),points2,points1)
        cv2.imwrite('./result1/final'+str(i)+'.jpg',output1)
        cv2.imwrite('./result2/final'+str(i)+'.jpg',output2)
        i = i+1
        k = cv2.waitKey(0)
        if k & 0xff==ord('q'):
            cv2.destroyAllWindows()
            break


    # plt.imshow(img_source)

if __name__ == '__main__':
    main()

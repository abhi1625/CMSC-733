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
import copy
def features(img,detector,predictor,resize_val):
    #initialize facial detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(resize_val,resize_val))
    img = cv2.resize(img,(resize_val,resize_val))
    rects = detector(gray,1)
    if np.shape(rects)[0] == 0:
        shape = 0
        feature_found = False
    else:
        feature_found = True
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for (i,rect) in enumerate(rects):
            # print('before shape')
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)
            # print('shape = ')
            # print(shape)

            (x,y,w,h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


            for (x,y) in shape:
                cv2.circle(img,(x,y),2,(0,0,255),-1)

    return img, shape, feature_found


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
    lam = 50
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
    print('faceswap begin...')
    #Define Face Detector and Predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor.dat')
    # /home/pratique/Downloads/cmsc733-Pictorial Information/project_2_face_swap/YourDirectoryID_p2
    #Setup input video
    cap1 = cv2.VideoCapture('/home/abhinav/CMSC-733/Abhi1625_p2/TestSet_P2/Test2.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

    # cap2 = cv2.VideoCapture('./Data/prateek.mp4')
    i=0
    while(True):
        ret1,img_src = cap1.read()
        print(ret1)
        # if not ret1:
            # continue
        print(img_src.shape)
        # ret2,img_tar = cap2.read()
        # if((not ret1) or (not ret2)):
            # break
        # print(ret)
        # plt.imshow(img_src)
        # plt.show()
        # print(img_src.dtype)
        # img_src = cv2.imread('/home/abhinav/CMSC-733/Abhi1625_p2/chahat_c.jpeg')
        # img_tar = cv2.imread('./Data/stephen.mp4')

        # shape1= np.shape(img_src)[:2]
        # shape2= np.shape(img_tar)[:2]
        # print('shape1 = '+str(shape1)+'shape2 = '+str(shape2))
        # a = np.minimum(shape1,shape2)
        # print(a)
        # a = np.minimum(a[0],a[1])
        # print('final dim  = '+str(a))
        rects = detector(img_src,2)
        img_source = copy.deepcopy(img_src)
        if len(rects)>1:
            img_target = img_source[rects[1].top()-40:rects[1].bottom()+40,rects[1].left()-40:rects[1].right()+40,:]
        else:
            cv2.imwrite('./result1/'+str(i)+'.jpg',img_src)
            i =  i+1
            print("2 faces not detected")
            continue
        # else:
            # img_target = cv2.imread('/home/abhinav/CMSC-733/Abhi1625_p2/imgs/joker.jpg')
        # img_source = cv2.resize(img_src,(200,200))
        # img_source = cv2.cvtColor(img_source,cv2.COLOR_BGR2RGB)
        # img_target = cv2.resize(img_tar,(200,200))
        # img_target = cv2.cvtColor(img_target,cv2.COLOR_BGR2RGB)
        img_source = img_source[rects[0].top()-40:rects[0].bottom()+40,rects[0].left()-40:rects[0].right()+40,:]
        img1 = img_source.copy()
        img2 = img_target.copy()



        a=200
        img_source = cv2.resize(img1.copy(),(a,a))
        # img_source = cv2.cvtColor(img_source,cv2.COLOR_BGR2RGB)
        img_target = cv2.resize(img2.copy(),(a,a))
        # img_target = cv2.cvtColor(img_target,cv2.COLOR_BGR2RGB)
        # img1 = img_src.copy()
        # img2 = img_tar.copy()

        img1,points1, flag1 = features(img_source,detector,predictor,a)
        img2,points2, flag2 = features(img_target,detector, predictor,a)
        if(flag1==False or flag2==False):
            cv2.imwrite('./result1/'+str(i)+'.jpg',img_src)
            i = i+1
            print('skipping frame, 68 features not found')
            continue

        output1 = swap(img_source.copy(),img_target.copy(),points1,points2)
        output2 = swap(img_target.copy(),img_source.copy(),points2,points1)
        print(output2.shape)
         t1 = cv2.resize(output1,((rects[0].right()+40)- (rects[0].left()-40),(rects[0].bottom()+40)-(rects[0].top()-40) ))
        output2 = cv2.resize(output2,((rects[1].right()+40)- (rects[1].left()-40),(rects[1].bottom()+40)-(rects[1].top()-40) ))

        # output2 = cv2.resize(output2,((rects[1].bottom()+40)-(rects[1].top()-40) ,(rects[1].right()+40)- (rects[1].left()-40)))

        # print(img_src.shape)
        img_src[rects[0].top()-40:rects[0].bottom()+40,rects[0].left()-40:rects[0].right()+40,:] = output1
        img_src[rects[1].top()-40:rects[1].bottom()+40,rects[1].left()-40:rects[1].right()+40,:] = output2

        cv2.imwrite('./result1/'+str(i)+'.jpg',img_src)
        # out.write(img_src)

        # cv2.imwrite('./result2/'+str(i)+'.jpg',output2)
        print('saved '+str(i)+' frames')
        i = i+1
        k = cv2.waitKey(0)
        if k & 0xff==ord('q'):
            cv2.destroyAllWindows()
            break
    cap1.release()
    out.release()
    print('operation ended...')

    # plt.imshow(img_source)

if __name__ == '__main__':
    main()

#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano

Author(s):

Abhinav Modi (abhi1625@terpmail.umd.edu)
Masters of Engineering in Robotics,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import numpy.linalg
import glob
import copy
import skimage.feature
import argparse
##ANMS

# def harrisFeatures(img):
#     img = img.deepcopy()
#     Rs = cv2.cornerHarris(img,2,3,k=0.04)
#     th = C<0.0001*C.max()
#     m = C>0.0001*C.max()
#     C[t] = 0
#     corners = np.where(m)
#
#     return Rs,corners

def ANMS(gray, num_corners):
    features = cv2.goodFeaturesToTrack(gray, 1000, 0.01,10)

    p,_,q = features.shape
    r = 1e+8 *np.ones([p,3])
    ED = 0

    for i in range(p):
        for j in range(p):
            xi = int(features[i,:,0])
            yi = int(features[i,:,1])
            xj = int(features[j,:,0])
            yj = int(features[j,:,1])
            if gray[yi,xi]>gray[yj,xj]:
                ED = (xj-xi)**2 +(yj-yi)**2
            if ED < r[i,0]:
                r[i,0] = ED
                r[i,1] = xi
                r[i,2] = yi
    feat = r[np.argsort(-r[:, 0])]
    best_corners = feat[:num_corners,:]
    return best_corners


# Create Feature Descriptor
def featdesc(img, pad_width,anms_out,patch_size):
    """
    img: grayscale image
    pad_width: width of padding on each of the 4 sides
    """
    feats = []
    if (patch_size%2) != 0:
        print('Patch Size should be even')
        return -1
    l,w = anms_out.shape
    img_pad = np.pad(img,(patch_size),'constant',constant_values=0)
    desc = np.array(np.zeros((int((patch_size/5)**2),1)))
    for i in range(l):
        patch = img_pad[int(anms_out[i][2]+(patch_size/2)):int(anms_out[i][2]+(3*patch_size/2)),int(anms_out[i][1]+(patch_size/2)):int(anms_out[i][1]+(3*patch_size/2))]

        blur_patch = cv2.GaussianBlur(patch,(5,5),0)

        sub_sample = blur_patch[0::5,0::5]
        cv2.imwrite('./patches/patch'+str(i)+'.png',sub_sample)
        feats = sub_sample.reshape(int((patch_size/5)**2),1)

        #make the mean 0
        feats = feats - np.mean(feats)

        #make the variance 1
        feats = feats/np.std(feats)
        cv2.imwrite('./features/feature_vector'+str(i)+'.png',feats)
        desc = np.dstack((desc,feats))

    return desc[:,:,1:]

# Match Pairs in the lists of two feature descriptors
def match_pairs(features1,features2,best_corners1,best_corners2):
    p,x,q = features1.shape
    m,y,n = features2.shape
    curr_min = 1e+8
    sec_min = 1e+8
    q = int(min(q,n))
    n = int(max(q,n))
    matchPairs = []
    # matchPairs2 = np.zeros([3,1])
    q = int(min(q,n))
    n = int(max(q,n))
    for i in range(q):
        match = {}
        for j in range(n):
            ssd = np.linalg.norm((features1[:,:,i]-features2[:,:,j]))**2
            match[ssd] = [best_corners1[i,:],best_corners2[j,:]]


        S = sorted(match)
        first = S[0]
        sec = S[1]
        if first/sec < 0.7:
            pairs = match[first]
            matchPairs.append(pairs)
    return matchPairs

# Draw matches for feature matching and Ransac
def showFeatures(img1,img2,matchPairs,new_img_name):
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    ################################################################################################
    r = 15
    thickness = 2
    c = None

    for i in range(len(matchPairs)):
        x1 = int(matchPairs[i][0][1])
        y1 = int(matchPairs[i][0][2])
        x2 = int(matchPairs[i][1][1])+int(img1.shape[1])
        y2 = int(matchPairs[i][1][2])

        cv2.line(new_img,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.circle(new_img,(x1,y1),3,255,-1)
        cv2.circle(new_img,(x2,y2),3,255,-1)

    cv2.imwrite(new_img_name,new_img)

#######################################################################################################
#Ransac for outlier rejection
def ransac(pairs, N,t,thresh):
    M = pairs
    H_new = np.zeros((3,3))
    max_inliers = 0
#     N = int(100*np.log(1-t)/np.log(1-(1-0.1)**4))
    # N = 1000
    for j in range(N):

        index = []
        pts = [np.random.randint(0,len(M)) for i in range(4)]
        p1 = np.array([[M[pts[0]][0][1:3]],[M[pts[1]][0][1:3]],[M[pts[2]][0][1:3]],[M[pts[3]][0][1:3]]],np.float32)
        p2 = np.array([[M[pts[0]][1][1:3]],[M[pts[1]][1][1:3]],[M[pts[2]][1][1:3]],[M[pts[3]][1][1:3]]],np.float32)

#         p1 = pts1[pts]
#         p2 = pts2[pts]
        H = cv2.getPerspectiveTransform( p1, p2 )
        inLiers = 0
        for ind in range(len(M)):
            source = np.array(M[ind][0][1:3])
            # print('source',source)
            target = np.array(M[ind][1][1:3])
            # print('target',target)
                                 #np.array([M[ind][1][1],M[ind][1][2]])
            predict = np.matmul(H, np.array([source[0],source[1],1]))
            # print('predict',predict)
            if predict[2] == 0:
                predict[2] = 0.000001
            predict_x = predict[0]/predict[2]
            predict_y = predict[1]/predict[2]
            predict = np.array([predict_x,predict_y])
            predict = np.float32([point for point in predict])
            if (np.linalg.norm(target-predict)) < thresh:
                inLiers += 1

                index.append(ind)
        pts1 = []
        pts2 = []
        if max_inliers < inLiers:
            max_inliers = inLiers
            [pts1.append([M[i][0][1:3]]) for i in index]
            [pts2.append([M[i][1][1:3]]) for i in index]
            # p1 =
            H_new,status = cv2.findHomography(np.float32(pts1),np.float32(pts2))
            if inLiers > t*len(M):
                print('success')


                break
    pairs = [M[i] for i in index]
    if len(pairs)<=4:
        print('Number of pairs after RANSAC is low')
    return H_new,pairs


def stitch_img(image, homography,image2_shape):
    '''
    image is the input image to be warped
    homography estimated using Ransac
    '''

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    h, w, z = image.shape

    # Find min and max x, y of new image
    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(homography, p)

    yrow = p_prime[1] / p_prime[2]
    xrow = p_prime[0] / p_prime[2]
    ymin = min(yrow)
    xmin = min(xrow)
    ymax = max(yrow)
    xmax = max(xrow)

    # Make new matrix that removes offset and multiply by homography
    new_mat = np.array([[1, 0, -1 * xmin], [0, 1, -1 * ymin], [0, 0, 1]])
    homography = np.dot(new_mat, homography)

    # height and width of new image frame
    height = int(round(ymax - ymin))+image2_shape[0]
    width = int(round(xmax - xmin))+ image2_shape[1]
    size = (height,width)
    # Do the warp
    warped = cv2.warpPerspective(src=image, M=homography, dsize=size)

    return warped, int(xmin), int(ymin)
###############################################################################################
def Estimated_Homography(img1,img2):
        # else:
        # img1 = images[im+1]
        # img2 = images[-1]
    flag = True
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray1 = np.float32(gray1)

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray2 = np.float32(gray2)

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    corners1 = cv2.goodFeaturesToTrack(gray1, 100000, 0.001,10)
    # corners1 = cv2.goodFeaturesToTrack(gray1, 10000, 0.0001,10)
    corners1 = np.int0(corners1)


    i1 = copy.deepcopy(img1)
    for corner in corners1:
        x,y = corner.ravel()
        cv2.circle(i1,(x,y),3,255,-1)
    cv2.imwrite('corners1.png',i1)

    corners2 = cv2.goodFeaturesToTrack(gray2, 100000, 0.001,10)
    # corners1 = cv2.goodFeaturesToTrack(gray1, 10000, 0.0001,10)
    corners2 = np.int0(corners2)


    i2 = copy.deepcopy(img2)
    for corner in corners2:
        x,y = corner.ravel()
        cv2.circle(i2,(x,y),3,255,-1)
    cv2.imwrite('corners2.png',i2)

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    best_corners1 = ANMS(gray1, 700)
    # anms1 = copy.deepcopy(img1)
    i1 = copy.deepcopy(img1)
    for corner1 in best_corners1:
        _,x1,y1 = corner1.ravel()
        cv2.circle(i1,(int(x1),int(y1)),3,255,-1)
    cv2.imwrite('anms1.png',i1)
    best_corners2 = ANMS(gray2, 700)
    # anms = copy.deepcopy(img2)
    i2 = copy.deepcopy(img2)
    for corner2 in best_corners2:
        _,x2,y2 = corner2.ravel()
        cv2.circle(i2,(int(x2),int(y2)),3,255,-1)
    cv2.imwrite('anms2.png',i2)
    """
    Feature Descriptors
    Save Feature Descriptor ay2output as FD.png
    """
    feat1 = featdesc(img=gray1, pad_width=40,anms_out=best_corners1,patch_size=40)
    feat2 = featdesc(img=gray2, pad_width=40,anms_out=best_corners2,patch_size=40)
    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    matchPairs = match_pairs(features1 = feat1,features2 = feat2, best_corners1 = best_corners1,best_corners2=best_corners2)
    print("Number of matches",len(matchPairs))
    if len(matchPairs)<45:
        print('Error')
        flag = False
    showFeatures(img1,img2,matchPairs,new_img_name = 'matching.png')
    """
    Refine: RANSAC, Estimate Homography
    """
    Hmg,pairs = ransac(matchPairs,N=3000,t=0.9 ,thresh=30.0)
    # showFeatures(img1,img2,matchPairs,new_img_name = 'matching.png')
    showFeatures(img1,img2,pairs,new_img_name = 'ransac.png')

    # Hmg = cv2.findHomography(matchPairs[],)
    return Hmg,flag


#Stitch and Blend the Panorama
def Blend(images):
    img1 = images[0]
    for im in images[1:]:
        # H = homography(img1, im, bff_match=False)
        H,flag = Estimated_Homography(img1,im)
        if flag == False:
            print('Number of matches is less than required')
            break
        # Hinv = np.linalg.inv(H)
        imgholder, origin_offset_x,origin_offset_y = stitch_img(img1,H,im.shape)
        oX = abs(origin_offset_x)
        oY = abs(origin_offset_y)
        for y in range(oY,im.shape[0]+oY):
            for x in range(oX,im.shape[1]+oX):

                img2_y = y - oY
                img2_x = x - oX
                imgholder[y,x,:] = im[img2_y,img2_x,:]

        img1 = imgholder
    # resize_pano = cv2.resize(img1,[1280,1024])
    # cv2.imwrite('pano.png',resize_pano)
    resize_pano = cv2.GaussianBlur(img1,(5,5),1.2)
    cv2.imwrite('blur_pano.png',resize_pano)
    return resize_pano

def main():
	# Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../Data/Train/Set2', help='Define Path of the Image Set folder')

    Args = Parser.parse_args()
    BasePath = Args.BasePath


    images = [cv2.imread(file) for file in sorted(glob.glob(str(BasePath)+'/*.jpg'))]


    mypano = Blend(images)
    print('number of images',len(images))


if __name__ == '__main__':
    main()

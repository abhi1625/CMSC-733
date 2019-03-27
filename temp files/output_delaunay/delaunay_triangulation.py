#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import dlib
from imutils import face_utils
import scipy.spatial as spatial




def draw_point(img, p, color ):
    cv2.circle( img, p, 2, color, cv2.cv.CV_FILLED, cv2.CV_AA, 0 )
def draw_features_on_img(img,detector,predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)

    if np.shape(rects)[0] == 0:
        shape = 0
        feature_found = False
    else:
        feature_found = True
#     print('rects')
#     print(rects)
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(img, rect)
            shape = face_utils.shape_to_np(shape)
            points = shape

    #         print(shape)
    #         print('next')

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    return img, shape, feature_found
# show the output image with the face detections + facial landmarks


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):

    triangleList = subdiv.getTriangleList();
    # print(triangleList)
    # print(len(triangleList))
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
    return img


# In[13]:


# print(len(chahat_points))


# In[14]:


# chahat_points[1]


# In[15]:


# np.shape(chahat_img)[2]


# In[16]:


def draw_matches(img1,pts1,img2,pts2):
    h = max(np.shape(img1)[0],np.shape(img2)[0])
    w = np.shape(img1)[1]+np.shape(img2)[1]
    w_shift = np.shape(img1)[1]
    new_img = np.hstack((img1,img2))
    for i,_ in enumerate(pts1):
        if i>19:
            cv2.line(new_img,(pts1[i][0],pts1[i][1]),(pts2[i][0]+w_shift,pts2[i][1]),(255,0,0),1)
    return new_img



# In[17]:


# montage = draw_matches(chahat_img.copy(), chahat_points, nitin_img.copy(), nitin_points)
# plt.rcParams["figure.figsize"] = (50,50)
# plt.imshow(montage)






# In[22]:


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
    left, top = np.min(feature_points, 0)
    # finding bottom right point in the image
    right, bottom = np.max(feature_points, 0)

    x, y = max(0, left-allowance), max(0, top-allowance)
    w, h = min(right+allowance, im_h)-x, min(bottom+allowance, im_w)-y
    new_points = feature_points - np.asarray([[x, y]])
    rect = (x, y, w, h)
    cropped_img = img[y:y+h, x:x+w]
    return new_points, rect, cropped_img



# In[25]:


def bilinear_interpolate(img, coords):
    """ Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T

def grid_coordinates(points):
    """ x,y grid coordinates within the ROI of supplied points
    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1
    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)

def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)


    return None


# In[26]:


def triangular_affine_matrices(vertices, src_points, dst_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dst_points to src_points
    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dst_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat


# In[27]:


def warp_image_3d(src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
    rows, cols = dst_shape[:2]
    result_img = np.zeros((rows, cols, 3), dtype=dtype)

    delaunay = spatial.Delaunay(dst_points)
#     print(delaunay.simplices)
#     print(len(delaunay.simplices))
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dst_points)))

    process_warp(src_img, result_img, tri_affines, dst_points, delaunay)

    return result_img


# plt.imshow(warped_src_face)


# In[29]:


def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel,iterations=1)

    return mask


# In[30]:


## Copy-and-paste
def apply_mask(img, mask):
    """ Apply mask to supplied image
    :param img: max 3 channel image
    :param mask: [0-255] values in mask
    :returns: new image with mask applied
    """
    masked_img=cv2.bitwise_and(img,img,mask=mask)

    return masked_img


# In[31]:


def correct_colours(im1, im2, landmarks1):
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur.astype(int)
    im2_blur += 128*(im2_blur <= 1)

    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result



def swap_faces(chahat_img, nitin_img,detector,predictor):
    chahat_img_size = np.shape(chahat_img)
    nitin_img_size = np.shape(nitin_img)
    shape1= chahat_img_size[:2]
    shape2= nitin_img_size[:2]
    print(shape1,shape2)
    a = np.minimum(np.asarray(shape1),np.asarray(shape2))
    a = np.minimum(a[0],a[1])
    # reshape_size = np.minimum(chahat_img_size,nitin_img_size)
    # print(tuple(reshape_size))
    # print(reshape_size)
    # if(reshape_size is chahat_img_size):
    nitin_img = cv2.resize(nitin_img,(a,a))
    #     print('nitin reshaped')
    # else:
    chahat_img = cv2.resize(chahat_img,(a,a))
        # print('chahat reshaped')

    nitin_img_copy =nitin_img.copy()

    chahat_features,chahat_points, flag1 = draw_features_on_img(chahat_img.copy(),detector,predictor)
    nitin_features,nitin_points, flag2 = draw_features_on_img(nitin_img.copy(),detector,predictor)

    if(flag1==False or flag2==False):
        # print('skipping frame, 68 features not found')
        features_found = False
        output = 0
        return output,features_found
    else:
        features_found = True

    plt.rcParams["figure.figsize"] = (10,10)

    size = np.shape(chahat_img)
    chahat_img_rect = (0,0,size[1],size[0])
    size = np.shape(nitin_img)
    nitin_img_rect = (0,0,size[1],size[0])

    chahat_img_subdiv  = cv2.Subdiv2D(chahat_img_rect)
    nitin_img_subdiv  = cv2.Subdiv2D(nitin_img_rect)

    chahat_points_tuple = tuple(map(tuple, chahat_points))

    for point in chahat_points_tuple:
        chahat_img_subdiv.insert(point)

    # print(np.shape(chahat_img))
    show_img = draw_delaunay(chahat_img.copy(), chahat_img_subdiv,(255, 255, 255));
    plt.imshow(show_img)


    # In[21]:


    nitin_points_tuple = tuple(map(tuple, nitin_points))
    for point in nitin_points_tuple:
        nitin_img_subdiv.insert(point)
    show_img = draw_delaunay(nitin_img.copy(), nitin_img_subdiv,(255, 255, 255));
    plt.imshow(show_img)

    src_points, src_shape, src_face = get_face_params(chahat_img.copy(), chahat_points)


    # In[24]:


    dst_points, dst_shape, dst_face = get_face_params(nitin_img.copy(), nitin_points)

    w, h = dst_face.shape[:2]
    # plt.imshow(src_face)
    warped_src_face = warp_image_3d(src_face.copy(), src_points, dst_points, (w, h))
    ## Mask for blending
    mask = mask_from_points((w, h), dst_points)
    # plt.imshow(mask)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask*mask_src, dtype=np.uint8)
    plt.imshow(warped_src_face)
    ## Correct color
    # if not args.warp_2d and args.correct_color:
    warped_src_face = apply_mask(warped_src_face, mask)
    dst_face_masked = apply_mask(dst_face, mask)
    plt.imshow(dst_face_masked)
    warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)

    plt.imshow(warped_src_face)

    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    ##Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

    x, y, w, h = dst_shape
    dst_img_cp = nitin_img_copy.copy()
    dst_img_cp[y:y+h, x:x+w] = output
    output = dst_img_cp

    return output, features_found


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    img_src = cv2.imread('./TestSet_P2/Scarlett.jpg')
    # chahat_img = cv2.cvtColor(chahat_img,cv2.COLOR_BGR2RGB)

    # img_dst = cv2.imread('./TestSet_P2/.jpg')
    # nitin_img = cv2.cvtColor(nitin_img,cv2.COLOR_BGR2RGB)

    i = 1
    # cap1 = cv2.VideoCapture('./TestSet_P2/Test1.mp4')
    cap2 = cv2.VideoCapture('./TestSet_P2/Test3.mp4')

    while(True):
        # ret1,img_src = cap1.read()
        ret2,img_tar = cap2.read()
        if((not ret2)):# or (not ret2)):
            break
        print('mean = '+str(np.mean(np.mean(img_src,axis =2))))
        output,flag = swap_faces(img_src, img_tar,detector,predictor)
        if(flag==False):# or flag2==False):
            # print(flag1)
            # print(flag2)
            cv2.imwrite('./delaunay_result1/'+str(i)+'.jpg',img_tar)
            i = i+1
            print('skipping frame, 68 features not found')
            continue

        cv2.imwrite('./delaunay_result1/'+str(i)+'.jpg',output)
        # cv2.imwrite('./result2/'+str(i)+'.jpg',output2)
        print('saved '+str(i)+' frames')
        i = i+1


if __name__ == '__main__':
    main()

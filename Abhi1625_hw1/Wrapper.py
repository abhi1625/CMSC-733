"""
CMSC733: Classical and Deep Learning Approaches for Computer Vision
Homework 1: AutoCalib: Camera Caliberation using Checkerboard

Author:
Abhinav Modi (abhi1625@umd.edu)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import glob
#Find Intrinsic parameters
def v(H,i,j):
    v = np.array([[H[0][i]*H[0][j]],
                  [H[0][i]*H[1][j] + H[0][j]*H[1][i]],
                  [H[1][i]*H[1][j]],
                  [H[2][i]*H[0][j] + H[0][i]*H[2][j]],
                  [H[2][i]*H[1][j] + H[1][i]*H[2][j]],
                  [H[2][i]*H[2][j]]])
    return np.reshape(v,(6,1))
def find_K(Homography):
    """
    Input H: 3D array with 3rd dimension as
    the number of Homography matrices observed
    """
    V_n = np.zeros([1,6])
    for i in range(Homography.shape[2]):

        V = np.vstack([v(Homography[:,:,i],0,1).T,
                      (v(Homography[:,:,i],0,0)-v(Homography[:,:,i],1,1)).T])
        # print(V.shape)

        V_n = np.concatenate([V_n,V],axis=0)
    # print(V_n.shape)
    V_n = V_n[1:]
    U,S,V = np.linalg.svd(V_n)
    # print(U)
    # print(V)
    # print(S)
    # print(V[5][5])
    b = V[:][5]
    # print(b)
    B = np.zeros([3,3])
    B[0][0]=b[0]
    B[0][1]=b[1]
    B[1][0]=b[1]
    B[0][2]=b[3]
    B[2][0]=b[3]
    B[1][1]=b[2]
    B[1][2]=b[4]
    B[2][1]=b[4]
    B[2][2]=b[5]
    # print(b[1])
    #Define K

    v0 = (B[0][1]*B[0][2] - B[0][0]*B[1][2])/(B[0][0]*B[1][1]-B[0][1]**2)

    lam = (B[2][2]-((B[0][2]**2)+v0*(B[0][1]*B[0][2] - B[0][0]*B[1][2]))/B[0][0])
    # print(lam)
    alp = math.sqrt(lam/B[0][0])
    beta = math.sqrt(((lam*B[0][0])/(B[0][0]*B[1][1] - B[0][1]**2)))
    # print(beta)
    gamma = -(B[0][1])*(alp**2)*beta/lam
    u0 = (gamma*v0/beta) - (B[0][2]*alp**2)/lam

    K = np.array([[alp, gamma, u0],
                  [0,   beta,  v0],
                  [0,   0,      1]])
    return K

def mat_params(K,H_image):
    Kinv = np.linalg.inv(K)
    Ah_hat = np.matmul(Kinv,H_image)  #K inv and columns of H
    lam = ((np.linalg.norm(np.matmul(Kinv,H_image[:,0]))+(np.linalg.norm(np.matmul(Kinv,H_image[:,1]))))/2)
    # print("lam", 1/lam)

    sgn = np.linalg.det(Ah_hat)
    if sgn<0:
        s = Ah_hat*-1/lam
    elif sgn>=0:
        s = Ah_hat/lam
    r1 = s[:,0]
    r2 = s[:,1]
    r3 = np.cross(r1,r2)
    t = s[:,2]
#     t = t/t[2]
    # print(r1)
    Q = np.array([r1,r2,r3]).T
    # print(Q)
    #Finding Rotation MAtrix from a 3x3 Matrix
    u,s,v =np.linalg.svd(Q)
    R = np.matmul(u,v)
    # print(R.shape)
    # print(t.shape)
    return np.hstack([R,np.reshape(t,(1,3)).T])

# def distortion(K, P, H_image, src_pts,dst_pts,k1,k2):
#
#     #Find m_hat
#     img_pts = np.matmul(np.matmul(K,P),src_pts)
#     u_hat = img_pts[0,:] + (img_pts[0,:] - K[0][2])*[(k1*((img_pts[0]/K[0][0])**2 + (img_pts[1]/K[1][1])**2)) +
#                                                 (k2*((img_pts[0]/K[0][0])**2 + (img_pts[1]/K[1][1])**2)**2)]
#     v_hat = img_pts[1,:] + (img_pts[1,:] - K[1][2])*[(k1*((img_pts[0]/K[0][0])**2 + (img_pts[1]/K[1][1])**2)) +
#                                                 (k2*((img_pts[0]/K[0][0])**2 + (img_pts[1]/K[1][1])**2)**2)]
#
#     print u_hat, v_hat

def parameters(A,k1,k2):
    a1 = np.reshape(np.array([A[0][0],0,A[0][2],A[1][1],A[1][2]]),(5,1))
    # a1 = np.reshape(A,(A.shape[0]*A.shape[1],1))
#     a2 = np.reshape(Rt,(Rt.shape[0]*Rt.shape[1],1))
    a3 = np.reshape(np.array([k1,k2]),(2,1))
    param = np.concatenate([a3,a1])
    return param


def fun(params,corners, Homographies):
#     corners=corner_points
#     Homographies=Homographies[:,:,1:]
    # A = np.reshape(params[2:11],(3,3))
    A = np.array([[params[2],0,params[4]],
                  [0,params[5],params[6]],
                  [0,0,1]])
    K = np.reshape(params[0:2],(2,1))
    w_xy = []
    for i in range(6):
        for j in range(9):
    #             print(j,i)
            w_xy.append([21.5*(j+1),21.5*(i+1),0,1])
    w_xyz = np.array(w_xy)

    error=np.empty([54,1])
    for i in range(Homographies.shape[2]):
        Rt = mat_params(A,Homographies[:,:,i])

        norm_pts = np.matmul(Rt,w_xyz.T)
        norm_pts = norm_pts/norm_pts[2]
        P = np.matmul(A,Rt)
        pt = np.matmul(P,w_xyz.T)
        img_pts = pt/pt[2]
        # if i ==0:
            # print("--------------------------")
            # print("pts",img_pts[:,0])
            # print("--------------------------")
        # print(img_pts[:,0]/img_pts[2,0])

        u_hat = img_pts[0] + (img_pts[0] - A[0][2])*[(K[0]*((norm_pts[0])**2 + (norm_pts[1])**2)) +
                                                        (K[1]*((norm_pts[0])**2 + (norm_pts[1])**2)**2)]
        v_hat = img_pts[1] + (img_pts[1] - A[1][2])*[(K[0]*((norm_pts[0])**2 + (norm_pts[1])**2)) +
                                                        (K[1]*((norm_pts[0])**2 + (norm_pts[1])**2)**2)]

        # mu_x = np.mean(img_pts[0])
        # mu_y = np.mean(img_pts[1])
        # var_x = math.sqrt(2/np.var(img_pts[0]))
        # var_y = math.sqrt(2/np.var(img_pts[1]))
        # u_hat = img_pts[0] + (img_pts[0] - A[0][2])*[(K[0]*(((img_pts[0] -mu_x)*var_x)**2 + ((img_pts[1]-mu_y)*var_y)**2)) +
        #                                                 (K[1]*(((img_pts[0] -mu_x)*var_x)**2 + ((img_pts[1] -mu_y)*var_y)**2)**2)]
        # v_hat = img_pts[1] + (img_pts[1] - A[1][2])*[(K[0]*(((img_pts[0] -mu_x)*var_x)**2 + ((img_pts[1] -mu_y)*var_y)**2)) +
        #                                                 (K[1]*(((img_pts[0] -mu_x)*var_x)**2 + ((img_pts[1] -mu_y)*var_y)**2)**2)]
        # u_hat = img_pts[0]
        # v_hat = img_pts[1]



        #
        # u_hat = img_pts[0] + (img_pts[0] - A[0][2])*[(K[0]*(((img_pts[0] - A[0][2])/np.amax(img_pts[0] - A[0][2]))**2 +
        #                                                 ((img_pts[1]-A[1][2])/np.amax(img_pts[1]-A[1][2]))**2)) +
        #                                                 (K[1]*(((img_pts[0]-A[0][2])/np.amax(img_pts[0]-A[0][2]))**2 +
        #                                                 ((img_pts[1]-A[1][2])/np.amax(img_pts[1]-A[1][2]))**2)**2)]
        # v_hat = img_pts[1] + (img_pts[1] - A[1][2])*[(K[0]*(((img_pts[0]-A[0][2])/np.amax(img_pts[0]-A[0][2]))**2 +
        #                                                 ((img_pts[1]-A[1][2])/np.amax(img_pts[1]-A[1][2]))**2)) +
        #                                                 (K[1]*(((img_pts[0]-A[0][2])/np.amax(img_pts[0]-A[0][2]))**2 +
        #                                                 ((img_pts[1]-A[1][2])/np.amax(img_pts[1]-A[1][2]))**2)**2)]

        proj = corners[i*54:(i+1)*54,0:2]
        proj = np.reshape(proj,(-1,2))
        reproj = np.reshape(np.array([u_hat,v_hat]),(2,54)).T
        # if i==0:
        #     print(reproj[0])
        #     print("proj, Reproj ",proj[0],reproj[0])
        # print('proj shape ', np.shape(proj))
        err = np.linalg.norm(np.subtract(proj,reproj),axis=1)**2
        # print("shapes ",np.subtract(proj,reproj).shape," ",err.shape)
        # print(err)

        error=np.vstack((error,err.reshape((54,1))))



    # print("error",error)
    # print("proj, Reproj ",proj[0],reproj[0])
    # print("#################################################")
    error=error[54:]
    error=np.reshape(error,(702,))
    # print(error)
    return error

def checkerboard_corners(img_list,world_xy):
    Homographies = np.zeros([3,3])
    corner_points = []
    for image in img_list:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        #Find Corners:
        ret, corners = cv2.findChessboardCorners(gray,(9,6),None)
        corner_points.extend(corners)
        # corners[0][0][0]

        cv2.circle(image,(corners[0][0][0],corners[0][0][1]),20,[255,0,255])
        #corresponding world_xy = [21.5,21.5]
        cv2.circle(image,(corners[8][0][0],corners[8][0][1]),20,[255,0,255])
        #corresponding world_xy = [193.5,21.5]
        cv2.circle(image,(corners[53][0][0],corners[53][0][1]),20,[255,0,255])
        #corresponding world_xy = [193.5,129]
        cv2.circle(image,(corners[45][0][0],corners[45][0][1]),20,[255,0,255])
        #corresponding world_xy = [21.5,129]
        img_dst= np.array([[corners[0][0]],
                           [corners[8][0]],
                           [corners[53][0]],
                           [corners[45][0]]],dtype='float32')
        H_image_world,status = cv2.findHomography(world_xy,img_dst)
        # H_image_world = cv2.getPerspectiveTransform(img_dst,world_xy)
        # print(H_image_world.shape)
        Homographies = np.dstack([Homographies,H_image_world])
        # warped_image = cv2.warpPerspective(image,H_image_world,(image.shape[1],image.shape[0]))
        # test = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
        # plt.imshow(image)
        # cv2.imshow('out',warped_image)
        # plt.show()
        # print(corners[0])

        # if cv2.waitKey(00)==ord('q'):
            # break/
        # print(corners)
    # print(Homographies.shape)

    corner_points = np.array(corner_points)
    return img_list, corner_points, Homographies[:,:,1:]
def RMSerror(A,K,Homographies,corner_points):

    w_xy = []
    for i in range(6):
        for j in range(9):
    #             print(j,i)
            w_xy.append([21.5*(j+1),21.5*(i+1),0])
    w_xyz = np.array(w_xy)

    # error=np.empty([54,1])
    mean = 0
    error = np.zeros([2,1])
    for i in range(Homographies.shape[2]):
        Rt = mat_params(A,Homographies[:,:,i])
        img_points,_ = cv2.projectPoints(w_xyz,Rt[:,0:3],Rt[:,3],A,K)
        img_points = np.array(img_points)
        # print(img_points.shape)
        errors = np.linalg.norm(corner_points[i*54:(i+1)*54,0,:]-img_points[:,0,:],axis=1)
        print(errors.shape)
        error = np.concatenate([error,np.reshape(errors,(errors.shape[0],1))])
        # error = np.mean(errors)
        # mean = error+mean
    mean_error = np.mean(error)
    return mean_error

def main():
    img_list = [cv2.imread(file) for file in glob.glob('./Calibration_Imgs/*.jpg')]
    clone_list=[]
    for img in img_list:
        clone_list.append(img.copy())

    world_xy = np.array([[21.5, 21.5],
                         [21.5*9,21.5],
                         [21.5*9, 21.5*6],
                         [21.5,21.5*6]], dtype='float32')

    img_list, corner_points, Homographies = checkerboard_corners(clone_list,world_xy)
    #Intrinsic Camera parameters
    A = find_K(Homographies)
    # print(A)
    params_init = parameters(A,0,0)
    res = least_squares(fun,x0=np.squeeze(params_init),method='lm',args=(corner_points,Homographies))
    # A = np.reshape(res.x[2:11],(3,3))
    A = np.array([[res.x[2],res.x[3],res.x[4]],
                  [0,res.x[5],res.x[6]],
                  [0,0,1]])
    K = np.reshape(res.x[0:2],(2,1))
    print("############################################")
    print("Final Intrinsic Camera matrix",A)
    print("Distortion parameters",K)
    # print("Cost",res.cost/(54*13))
    print("############################################")

    #Undistort Image13
    distortion = np.array([K[0],K[1],0,0,0],dtype=float)
    undist_images = []
    for image in clone_list:
        undist = cv2.undistort(image,A,distortion)
        undist_images.append(undist)

    undist_list, undist_corners,_ = checkerboard_corners(undist_images,world_xy)
    reproj_error = RMSerror(A,distortion,Homographies,corner_points)
    # rmsError = np.linalg.norm(corner_points[:,0,:] - undist_corners[:,0,:],axis=1)
    # rmsError = np.sqrt(np.mean(np.linalg.norm(corner_points[:,0,:] - undist_corners[:,0,:],axis=1))**2)
    print("RMS Error",reproj_error)
    # i = 0
    # for image in undist_list:
    #     for j in range(i*54,(i+1)*54):
    #         cv2.circle(image,(corner_points[j,0,0],corner_points[j,0,1]),20,[255,0,0],1)
    #         cv2.circle(image,(undist_corners[j,0,0],undist_corners[j,0,1]),20,[255,0,255],1)
    #     # cv2.drawChessboardCorners(image,(9,6),corner_points[i*54:(i+1)*54,0,:],True)
    #     # cv2.drawChessboardCorners(image,(9,6),undist_corners[i*54:(i+1)*54,0,:],True)
    #     cv2.imwrite(str(i)+".png",image)
    #     cv2.imshow("test",image)
    #     while(True):
    #         if cv2.waitKey(00)==ord('q'):
    #             cv2.destroyAllWindows()
    #             break
    #     i = i+1
    #For Image 1 find the extrinsic parameters
    # Rt = mat_params(A,Homographies[:,:,1])

if __name__ == '__main__':
    main()
# w_xy = []
# for i in range(6):
#     for j in range(9):
# #             print(j,i)
#         w_xy.append([21.5*(j+1),21.5*(i+1),0,1])
# w_xyz = np.array(w_xy)
# w_xyz
# Rt = mat_params(A,Homographies[:,:,1])
# P = np.matmul(A,Rt)
# pt = np.matmul(P,w_xyz.T)
# pt/pt[2]

#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s):
Abhinav Modi (abhi1625@umd.edu)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.stats as st
import skimage.transform
import sklearn.cluster
import argparse



def gauss2D(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""
#     nsig = scales*scales
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def gaussian1d(sigma, mean, x, ord):
    x = np.array(x)
    x_ = x - mean
    var = sigma**2

    # Gaussian Function
    g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))

    if ord == 0:
        g = g1
        return g
    elif ord == 1:
        g = -g1*((x_)/(var))
        return g
    else:
        g = g1*(((x_*x_) - var)/(var**2))
        return g

def gaussian2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return g

def log2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    h = g*((x*x + y*y) - var)/(var**2)
    return h

def makefilter(scale, phasex, phasey, pts, sup):

    gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
    gy = gaussian1d(scale,   0, pts[1,...], phasey)

    image = gx*gy

    image = np.reshape(image,(sup,sup))
    return image


#Define LM filters
def makeLMfilters(sup, scales, norient, nrotinv):
    scalex  = np.sqrt(2) * np.arange(1,scales+1)
    nbar  = len(scalex)*norient
    nedge = len(scalex)*norient
    nf    = nbar+nedge+nrotinv
    F     = np.zeros([sup,sup,nf])
    hsup  = (sup - 1)/2

    x = [np.arange(-hsup,hsup+1)]
    y = [np.arange(-hsup,hsup+1)]

    [x,y] = np.meshgrid(x,y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts =	 np.array(orgpts)

    count = 0
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c+0,-s+0],[s+0,c+0]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts,orgpts)
            F[:,:,count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
            F[:,:,count+nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
            count = count + 1

    count = nbar+nedge
    scales = np.sqrt(2) * np.array([1,2,3,4])

    for i in range(len(scales)):
        F[:,:,count]   = gaussian2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, 3*scales[i])
        count = count + 1

    return F

#Define DOG Filters
def makeDOGFilters(scales,orient,size):
    kernels=[]
    for scale in scales:
        orients=np.linspace(0,360,orient)
        kernel=gauss2D(size,scale)
        border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
        sobelx64f = cv2.Sobel(kernel,cv2.CV_64F,1,0,ksize=3, borderType=border)
        for i,eachOrient in enumerate(orients):
            #plt.figure(figsize=(16,16))
            image=skimage.transform.rotate(sobelx64f,eachOrient)
            #plt.subplots_adjust(hspace=0.1,wspace=1.5)
            #plt.subplot(scales,orient,i+1)
            #plt.imshow(image,cmap='binary')
            kernels.append(image)
            image=0
    return kernels


def gabor_fn(sigma, theta, Lambda, psi, gamma):
    """
    sigma is the variance
    theta is the orientation
    lambda is the wavelength of the sinusoidal carrier
    psi =
    """
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
#     xmax = 15
#     ymax = 15
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
#     plt.imshow(gb,cmap='binary')
    return gb
#4, 0.25, 1, 1.0, 1

#Define Gabor Filters
def makeGaborFilters(sigma, theta, Lambda, psi, gamma,num_filters):
    g = list()
    for j in sigma:
        gb = gabor_fn(j, theta, Lambda, psi, gamma)
        ang = np.linspace(0,360,num_filters)
        for i in range(num_filters):
            image = skimage.transform.rotate(gb,ang[i])
            g.append(image)
    return g

#Define Texton Map by using DOG filters
def texton_DOG(Img, filter_bank):
    tex_map = np.array(Img)
#     _,_,num_filters = filter_bank.shape
    num_filters = len(filter_bank)
    for i in range(num_filters):
#         out = cv2.filter2D(img,-1,filter_bank[:,:,i])
        out = cv2.filter2D(Img,-1,filter_bank[i])
        tex_map = np.dstack((tex_map, out))
    return tex_map

#Define Texton Map using LM filters
def texton_LM(Img, filter_bank ):
    tex_map = np.array(Img)
    _,_,num_filters = filter_bank.shape
#     num_filters = len(filter_bank)
    for i in range(num_filters):
        out = cv2.filter2D(Img,-1,filter_bank[:,:,i])
#         out = cv2.filter2D(img,-1,filter_bank[i])
        tex_map = np.dstack((tex_map, out))
    return tex_map

def Texton(img,filter_bank1,filter_bank2,filter_bank3, num_clusters):
    p,q,_ = img.shape
    tex_map_DOG = texton_DOG(img, filter_bank2)
    tex_map_LM = texton_LM(img, filter_bank1)
    tex_map_Gabor = texton_DOG(img, filter_bank3)
    tex_map = np.dstack((tex_map_DOG[:,:,1:],tex_map_LM[:,:,1:],tex_map_Gabor[:,:,1:]))
    m,n,r = tex_map.shape
    inp = np.reshape(tex_map,((p*q),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = num_clusters, random_state = 2)
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    l = np.reshape(labels,(m,n))
    plt.imshow(l)
    return l

def brightness(Img, num_clusters):
    p,q,r = Img.shape
    inp = np.reshape(Img,((p*q),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = num_clusters, random_state = 2)
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    l = np.reshape(labels,(p,q))
    plt.imshow(l,cmap = 'binary')
    return l

def color(Img, num_clusters):
    p,q,r = Img.shape
    inp = np.reshape(Img,((p*q),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = num_clusters, random_state = 2)
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    l = np.reshape(labels,(p,q))
    plt.imshow(l)
    return l

def gradient(Img, bins, filter_bank):
    gradVar = Img
    for N in range(len(filter_bank)/2):
        g = chi_sqr_gradient(Img, bins, filter_bank[2*N],filter_bank[2*N+1])
        gradVar = np.dstack((gradVar,g))
    mean = np.mean(gradVar,axis =2)
    return mean
#Define half disk filters for gradient calculation
def half_disk(radius):
    a=np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    mask2 = x*x + y*y <= radius**2
    a[mask2] = 0
    b=np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    p = x>-1
    q = y>-radius-1
    mask3 = p*q
    b[mask3] = 0

    return a, b

def disk_masks(scales, orients):
    flt = list()
    orients = np.linspace(0,360,orients)
    for i in scales:
        radius = i
        g = list()
        a,b = half_disk(radius = radius)

        for i,eachOrient in enumerate(orients):
            c1 = skimage.transform.rotate(b,eachOrient,cval =1)
            z1 = np.logical_or(a,c1)
            z1 = z1.astype(np.int)
            b2 = np.flip(b,1)
            c2 = skimage.transform.rotate(b2,eachOrient,cval =1)
            z2 = np.logical_or(a,c2)
            z2 = z2.astype(np.int)
            flt.append(z1)
            flt.append(z2)
    # for each in flt:
    #     plt.imshow(each,cmap='binary')
    #     plt.show()

    return flt
def chi_sqr_gradient(Img, bins,filter1,filter2):
    chi_sqr_dist = Img*0
    g = list()
    h = list()
    for i in range(bins):
        #numpy.ma.masked_where(condition, a, copy=True)[source]
        #Mask an array where a condition is met.
        img = np.ma.masked_where(Img == i,Img)
        img = img.mask.astype(np.int)
        g = cv2.filter2D(img,-1,filter1)
        h = cv2.filter2D(img,-1,filter2)
        chi_sqr_dist = chi_sqr_dist + ((g-h)**2 /(g+h))
    return chi_sqr_dist/2
def plot_LM(filters):
    _,_,r = filters.shape
    plt.subplots(4,12,figsize=(20,20))
    for i in range(r):
        plt.subplot(4,12,i+1)
        plt.axis('off')
        plt.imshow(filters[:,:,i],cmap='binary')
    plt.savefig('LM.png')
    plt.close()
        # x = filters[:,:,i]
        # border = cv2.copyMakeBorder(x,10,10,10,10,cv2.BORDER_CONSTANT,value = [255,255,255])
    #     # fig = (border,) + fig
    # return cv2.hconcat(fig)
def plot_Gab(filters):
    r = len(filters)
    plt.subplots(r/5,5,figsize=(20,20))
    for i in range(r):
        plt.subplot(r/5,5,i+1)
        plt.axis('off')
        plt.imshow(filters[i],cmap='gray')
    plt.savefig('Gabor.png')
    plt.close()

def plot_DoG(filters):
    r = len(filters)
    plt.subplots(r/5,5,figsize=(20,20))
    for i in range(r):
        plt.subplot(r/5,5,i+1)
        plt.axis('off')
        plt.imshow(filters[i],cmap='gray')
    plt.savefig('DoG.png')
    plt.close()
def plot_halfdisks(filters):
    r = len(filters)
    plt.subplots(r/5,5,figsize=(20,20))
    for i in range(r):
        plt.subplot(r/4,4,i+1)
        plt.axis('off')
        plt.imshow(filters[i],cmap='binary')
    plt.savefig('Half-Disks.png')
    plt.close()

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Maps_flag', default=False)
    Args = Parser.parse_args()
    Maps_flag = Args.Maps_flag


    for i in range(10):
        path = '/home/abhinav/CMSC-733/Abhi1625_hw0/Phase1/BSDS500/Images/'+str(i+1)+'.jpg'
        print(path)
        img = plt.imread(path)  #0 for reading img in grayscale
        img_col = plt.imread('/home/abhinav/CMSC-733/Abhi1625_hw0/Phase1/BSDS500/Images/'+str(i+1)+'.jpg')
        # img = cv2.cvtColor(img,)

        """
        Generate Leung-Malik Filter Bank: (LM)
        Display all the filters in this filter bank and save image as LM.png,
        use command "cv2.imwrite(...)"
        """
        filter_bank1 = makeLMfilters(sup = 49, scales = 3, norient = 6, nrotinv = 12)
        plot_LM(filter_bank1)
        # cv2.imwrite('LM.png',flt1)
        """
        Generate Difference of Gaussian Filter Bank: (DoG)
        Display all the filters in this filter bank and save image as DoG.png,
        use command "cv2.imwrite(...)"
        """
        filter_bank2 = makeDOGFilters(scales = [9,16,25],orient = 15, size = 49 )
        plot_DoG(filter_bank2)
        # cv2.imwrite('DoG.png',flt2)


        """
        Generate Gabor Filter Bank: (Gabor)
        Display all the filters in this filter bank and save image as Gabor.png,
        use command "cv2.imwrite(...)"
        """
        filter_bank3 = makeGaborFilters(sigma=[9,16,25], theta=0.25, Lambda=1, psi=1, gamma=1,num_filters=15)
        plot_Gab(filter_bank3)

        if(Maps_flag):
            """
            Generate texture ID's using K-means clustering
            Display texton map and save image as TextonMap_ImageName.png,
            use command "cv2.imwrite('...)"
            """
            T = Texton(img_col,filter_bank1,filter_bank2,filter_bank3,num_clusters=64)
            np.save('Maps/T'+str(i+1),T)
            plt.imsave(str(i+1)+"/TextonMap_"+str(i+1)+".png", T)

            """
            Generate Brightness Map
            Perform brightness binning
            """
            B = brightness(Img = img, num_clusters=16)
            np.save('Maps/B'+str(i+1),B)
            plt.imsave(str(i+1)+"/BrightnessMap_"+str(i+1)+".png", B,cmap='binary')

            """
            Generate Color Map
            Perform color binning or clustering
            """
            C = color(img_col, 16)
            np.save('Maps/C'+str(i+1),C)
            plt.imsave(str(i+1)+"/ColorMap_"+str(i+1)+".png", C)

        else:
            T = np.load('Maps/T'+str(i+1)+'.npy')
            B = np.load('Maps/B'+str(i+1)+'.npy')
            C = np.load('Maps/C'+str(i+1)+'.npy')

        """
        Generate Half-disk masks
        Display all the Half-disk masks and save image as HDMasks.png,
        use command "cv2.imwrite(...)"
        """
        c = disk_masks([5,7,16], 8)
        plot_halfdisks(c)
        """
        Generate Texton Gradient (Tg)
        Perform Chi-square calculation on Texton Map
        Display Tg and save image as Tg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Tg = gradient(T, 64, c)
        plt.imsave(str(i+1)+"/Tg_"+str(i+1)+".png", Tg)

        # plt.imshow(Tg)
        # plt.show()
        """
        Generate Brightness Gradient (Bg)
        Perform Chi-square calculation on Brightness Map
        Display Bg and save image as Bg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Bg = gradient(B, 16, c)
        plt.imsave(str(i+1)+"/Bg_"+str(i+1)+".png", Bg,cmap='binary')

        """
        Generate Color Gradient (Cg)
        Perform Chi-square calculation on Color Map
        Display Cg and save image as Cg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Cg = gradient(C, 16, c)
        plt.imsave(str(i+1)+"/Cg_"+str(i+1)+".png", Cg)


        temp = (Tg+Bg+Cg)/3
        # cv2.imshow('temp',temp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        """
        Read Sobel Baseline
        use command "cv2.imread(...)"
        """
        sobelBaseline = plt.imread('/home/abhinav/CMSC-733/Abhi1625_hw0/Phase1/BSDS500/SobelBaseline/'+str(i+1)+'s.png',0)

        """
        Read Canny Baseline
        use command "cv2.imread(...)"
        """
        cannyBaseline = plt.imread('/home/abhinav/CMSC-733/Abhi1625_hw0/Phase1/BSDS500/CannyBaseline/'+str(i+1)+'.png',0)

        """
        Combine responses to get pb-lite output
        Display PbLite and save image as PbLite_ImageName.png
        use command "cv2.imwrite(...)"
        """
        pblite_out = np.multiply(temp, (0.1*cannyBaseline+0.9*sobelBaseline))

        cv2.imwrite(str(i+1)+"/PbLite_" + str(i+1) + "canny=0.1.png", pblite_out)


    # plt.imshow(pblite_out,cmap='binary')
    # plt.show()

if __name__ == '__main__':
    main()

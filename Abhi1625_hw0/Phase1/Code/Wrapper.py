#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s):
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu)
PhD Student in Computer Science,
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



def makeLMfilters(sup, scales, norient, nrotinv):
#     sup     = 49
    scalex  = np.sqrt(2) * np.arange(1,scales+1)
#     norient = 6
#     nrotinv = 12

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
	#
    # for i in range(len(scales)):
    #     F[:,:,count]   = gaussian2d(sup, scales[i])
    #     count = count + 1
	#
    # for i in range(len(scales)):
    #     F[:,:,count] = log2d(sup, scales[i])
    #     count = count + 1
	#
    # for i in range(len(scales)):
    #     F[:,:,count] = log2d(sup, 3*scales[i])
    #     count = count + 1
	#
    # return F


def gauss2D(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""
#     nsig = scales*scales
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel
def makeDOGFilters(scales,orient,size):
    orients=np.linspace(0,360,orient)
    kernels=[]
    kernel=gauss2D(size,scales)
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

def makeGaborFilters(sigma, theta, Lambda, psi, gamma,num_filters):
    gb = gabor_fn(sigma, theta, Lambda, psi, gamma)
    g = list()
    ang = np.linspace(0,360,num_filters)
    for i in range(num_filters):
        image = skimage.transform.rotate(gb,ang[i])
        g.append(image)
    return g

def texton_DOG(Img, filter_bank):
    tex_map = np.array(Img)
#     _,_,num_filters = filter_bank.shape
    num_filters = len(filter_bank)
    for i in range(num_filters):
#         out = cv2.filter2D(img,-1,filter_bank[:,:,i])
        out = cv2.filter2D(Img,-1,filter_bank[i])
        tex_map = np.dstack((tex_map, out))
    return tex_map
def texton_LM(Img, filter_bank ):
    tex_map = np.array(Img)
    _,_,num_filters = filter_bank.shape
#     num_filters = len(filter_bank)
    for i in range(num_filters):
        out = cv2.filter2D(Img,-1,filter_bank[:,:,i])
#         out = cv2.filter2D(img,-1,filter_bank[i])
        tex_map = np.dstack((tex_map, out))
    return tex_map

def clustering(img,filter_bank1,filter_bank2,filter_bank3, num_clusters):
    p,q = img.shape
    tex_map_DOG = texton_DOG(img, filter_bank2)
    tex_map_LM = texton_LM(img, filter_bank1)
    tex_map_Gabor = texton_DOG(img, filter_bank3)
    tex_map = np.dstack((tex_map_DOG,tex_map_LM,tex_map_Gabor))
    m,n,r = tex_map.shape
    inp = np.reshape(tex_map,((p*q),r))
    kmeans = sklearn.cluster.KMeans(n_clusters = 64, random_state = 2)
    kmeans.fit(inp)
    labels = kmeans.predict(inp)
    l = np.reshape(labels,(m,n))
    plt.imshow(l)
    return l


def brightness(Img, num_clusters):
    p,q = Img.shape
    inp = np.reshape(Img,((p*q),1))
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
    return gradVar
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
scales = [5,7,9]
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
    for each in flt:
        plt.imshow(each,cmap='binary')
        plt.show()

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


def main():
	img = cv2.imread('/home/abhinav/CMSC-733/Abhi1625_hw0/Phase1/BSDS500/Images/1.jpg',0)  #0 for reading img in grayscale
	img_col = cv2.imread('/home/abhinav/CMSC-733/Abhi1625_hw0/Phase1/BSDS500/Images/1.jpg')  #0 for reading img in grayscale

	filter_bank1 = makeLMfilters(sup = 49, scales = 3, norient = 6, nrotinv = 12)
	filter_bank2 = makeDOGFilters(scales = 16,orient = 15, size = 49 )
	filter_bank3 = makeGaborFilters(sigma=4, theta=0.25, Lambda=1, psi=1, gamma=1,num_filters=15)



	T = clustering(img,filter_bank1,filter_bank2,filter_bank3,num_clusters=64)
	B = brightness(Img = img, num_clusters=16)
	C = color(img_col, 16)
	c = disk_masks([7], 4)


	Tg = gradient(T, 64, c)

	Bg = gradient(B, 16, c)

	Cg = gradient(C, 16, c)



	temp = (Tg+Bg+Cg)/3
	mean = np.mean(temp,axis =2)
	cannyBaseline = cv2.imread('/home/abhinav/CMSC-733/Abhi1625_hw0/Phase1/BSDS500/CannyBaseline/1.png',0)
	sobelBaseline = cv2.imread('/home/abhinav/CMSC-733/Abhi1625_hw0/Phase1/BSDS500/SobelBaseline/1.png',0)
	final = np.multiply(mean, (0.3*cannyBaseline+0.7*sobelBaseline))

	plt.imshow(final,cmap='binary')
	plt.show()
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""



	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""


	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Brightness Map
	Perform brightness binning
	"""


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Color Map
	Perform color binning or clustering
	"""


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""

if __name__ == '__main__':
    main()

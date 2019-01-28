import numpy as np
import scipy.stats as st
import skimage.transform
import matplotlib.pyplot as plt
import cv2

def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel
def main():

    kernel=gkern(21,7)
    border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
    sobelx64f = cv2.Sobel(kernel,cv2.CV_64F,1,0,ksize=5, borderType=border)
    sobely64f = cv2.Sobel(kernel,cv2.CV_64F,0,1,ksize=5, borderType=border)
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    final=skimage.transform.rotate(sobelx64f,90)
    plt.imshow(final,cmap='binary')
if __name__ == '__main__':
    main()
